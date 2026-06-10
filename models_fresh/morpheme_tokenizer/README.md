# morpheme_tokenizer — deployable production tokenizer (2026-06-10)

The morpheme-aware BPE turned into a **real, deployable LLM tokenizer**: fixed 64k integer
vocab, encode/decode to ids, **guaranteed-lossless** round-trip, and a fast lookup path that
avoids the analyzer for known words. Code lives in the `turkish-corpus` repo
(`turkish_corpus.morpheme_tokenizer.MorphemeTokenizer`, commit `d7ce7a4`); this directory is
the trained artifact.

## Design

- **64k vocab:** PAD/UNK/BOS/EOS + `WORD_BOUNDARY` + `CAP`/`UPPER` casing markers + 256
  GPT-2 byte-chars + **16k byte-BPE merge pieces** + the rest morpheme pieces (▁-fused
  word-initial + bare internal, frequency-ranked).
- **Fused word boundary** (`▁gel` is one token, not `▁`+`gel`) — no per-word boundary penalty.
- **Byte-BPE OOV fallback:** a word that can't be cleanly morpheme-encoded is byte-BPE'd on its
  *verbatim* surface — a few sub-word tokens, not a raw-byte explosion, and exact bytes.
- **Turkish casing markers** + a **fidelity-checked** encode: the clean morpheme path is used
  only when the analyzed pieces reconstruct the lowercased surface AND are all in-vocab;
  otherwise byte-BPE the original surface. → `decode(encode(text)) == text` for ANY input
  (casing, apostrophes, foreign, emoji).
- **Fast `word→pieces` table** (593k entries) + per-word `lru_cache` + lazy tr_api fallback.

## Build inputs

- `../morpheme_bpe_prod.json` — 30k morpheme BPE merges trained on the **word-based** morph
  data (`/tmp/morph20k_fast.txt`, 20k lines / 6.1M words / 594k unique), consistent with the
  per-word encode path.
- Byte-BPE: 16k merges over the top-40k surfaces (GPT-2 byte-chars).
- Built with `turkish-corpus/scripts/build_morpheme_tokenizer.py`.

## Held-out validation (eval.txt; 8000 words / 500 lines)

| Metric | Value |
|--------|------:|
| **Lossless round-trip** | **500/500 lines exact** ✅ (casing + apostrophes preserved) |
| Blended fertility | **1.846** |
| — idealized morpheme pieces (no vocab cap) | 1.082 |
| — table-HIT words (75.5%) | 1.238 |
| — table-MISS tail (24.5%, via byte-BPE) | 3.718 |
| raw single-byte tokens | 12.6% (was 47.4% before byte-BPE) |
| encode throughput (table) | ~1,256 words/s (misses hit tr_api) |
| *ref:* sp_unigram_32000 | 1.465 |
| *ref:* morpheme_bpe_20000 (idealized, no vocab/OOV) | 1.158 |

## Honest status

- **Fidelity: solved.** Lossless round-trip including capitalization and apostrophes — the two
  things tr_api normalization destroyed. This is production-safe.
- **Fertility: not yet winning.** The morpheme advantage is real (1.082 at the piece level),
  but the blended **1.846 is still above sp_unigram (1.465)** because the morpheme table built
  from 20k lines covers only **75.5%** of held-out words; the **24.5% OOV tail costs ~3.7
  tokens/word** via byte-BPE and dominates the blend.
- **The lever is table/morpheme coverage**, which scales with how much text is segmented:
  - ~92% coverage → blended ≈ 1.49 (≈ sp_unigram)
  - ~95% coverage → blended ≈ 1.37 (beats sp_unigram)
  - 75.5% (now) → 1.846
  Reaching ~95% needs ~100k–1M+ segmented lines via the hardened segmenter
  (`turkish-corpus` `morph_segment.py` / `segment_morphemes_fast.py`) — **cluster-scale work**,
  matching the plan to run the heavy segmentation on the Linux/GPU box.

**Bottom line:** the architecture + fidelity are done and deployable today; beating sp_unigram
on fertility is a data-scale step (bigger morpheme table), not a design change.

## Files
- `tokenizer.json` — vocab, morpheme merges, byte-BPE merges, casing/byte sections.
- `table.tsv` — `word⇥piece1▁piece2…` fast table (593k entries).
- Load: `MorphemeTokenizer.load("models_fresh/morpheme_tokenizer", repo_path=<turkish-tokenizer>)`.
