# models_fresh_scaled — sp_morph scaling experiment (2026-06-10)

Does training the deployable morpheme-aware tokenizer (`sp_morph` = SentencePiece Unigram on
morpheme-segmented text, **no analyzer at inference**) on **more data** lower its fertility?

Short answer: **barely.** The apparent big improvement was a *segmentation-method* artifact,
not data scale. `sp_morph` fertility plateaus around **~1.5** by ~5k lines.

## Hardened segmenter

Scaling was previously blocked because turkish-llm's `segment_morphemes.py` (`mp.Pool.imap`,
~115 w/s, no timeout) hung on pathological tokens and orphaned workers for hours. The
sibling **turkish-corpus** repo now has a hardened replacement
(`src/turkish_corpus/morph_segment.py`, `scripts/segment_morphemes_fast.py`): **per-word
SIGALRM timeout** (pathological tokens can't hang — pass through as one token), **per-word
cache** (Turkish ~5.5% type/token → ~18× fewer analyzer calls), **>70-char passthrough**
(URL/junk never hits tr_api), and **clean pool shutdown** (no orphans). 20k lines = 706,395
unique words segmented at ~555 w/s in ~21 min, clean.

## Results — held-out eval (8000 words; lower fertility = better)

| Setup | seg. method | fertility | tok/morph | boundary-F1 |
|-------|-------------|----------:|----------:|------------:|
| sp_morph_32000, 5k | **line-based** (segment_morphemes.py) | 2.039 | 0.789 | 0.657 |
| sp_morph_32000, 5k | **word-based** (hardened tool) — *control* | **1.552** | 0.812 | 0.432 |
| sp_morph_32000, 20k | **word-based** (hardened tool) — *4× data* | **1.507** | 0.789 | 0.430 |
| sp_morph_16000, 20k | word-based | 1.627 | 0.852 | 0.505 |
| sp_morph_48000, 20k | word-based | 1.454 | 0.761 | 0.385 |
| *ref:* sp_unigram_32000 | (raw text) | 1.465 | 0.767 | 0.248 |
| *ref:* morpheme_bpe_20000 | (analyzer + merges) | 1.158 | 0.606 | 0.270 |

Reports: `reports/eval_20260610T073847Z.*` (20k), control eval logged in the run output.

## Findings

1. **The 2.04 → 1.51 drop was the METHOD, not the data.** Holding the segmenter fixed
   (word-based), 4× more data moved sp_morph_32000 only **1.552 → 1.507 (~3%)** and
   boundary-F1 essentially not at all. The earlier line-based 5k (2.039) over-fragments the
   morph training text (line-level `tokenize_text` + clitic splitting → shorter learned
   pieces → higher fertility); the word-based per-word segmentation (`split_clitics=False`,
   `▁`-joined morphemes) yields lower fertility on its own.
2. **sp_morph fertility plateaus ~1.5.** More data barely helps; larger vocab is the lever
   (48k → 1.454). At its best it **ties plain sp_unigram (~1.465)** and stays **well above
   morpheme_bpe (1.158)**.
3. **Alignment ↔ compression tradeoff confirmed.** The word-based sp_morph is less
   morpheme-aligned (boundary-F1 ~0.43) than the line-based one (0.66) but much lower
   fertility. You don't get both from sp_morph.

## Takeaway for the LLM tokenizer

The *deployable* morpheme-aware tokenizer (`sp_morph`, no runtime analyzer) tops out near
plain-SentencePiece compression (~1.45–1.5) regardless of training-data scale. To beat that,
you need `morpheme_bpe` (1.158) — which requires the tr_api analyzer at inference (mitigated
by the per-word cache + a precomputed top-1M `word→pieces` table; see the turkish-corpus
`docs/tokenizer.md`). Net: it's a real choice between **morpheme_bpe (best compression, needs
analyzer)** and **sp_morph/sp_unigram (deployable, ~1.5 fertility)**.

## Files
- `sp_morph_{16000,32000,48000}.{model,vocab}` — trained on 20k word-based morph lines.
- Control (5k word-based, sp_morph_32000) lives in `../models_ctl5k/` for the method comparison.
