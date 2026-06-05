# Experiment spec: morphology-aware tokenizer A/B

**Question.** Holding architecture, training *text*, and compute fixed, does a
morphology-aware tokenizer (**morpheme-BPE**, ~48k vocab, fertility 1.30,
boundary precision 1.0) produce a *better* small Turkish LM than the standard
**SentencePiece-Unigram 32k** baseline?

This is the experiment that turns the tokenizer-level Pareto result
(`reports/latest.md`) into a claim about modeling. Decisions below reflect:
**both fairness axes**, **~125M model**, **from-scratch PyTorch** (chosen over
HF `Trainer` for full control and trivial integration of the analyzer-backed
morpheme-BPE encoder; we borrow nanoGPT patterns).

## 0. Claim shape

"Better" is decomposed, because the two tokenizers are **not** comparable on
per-token loss (a morpheme-BPE token spans more text than a Unigram token):

1. **Language modeling** — lower **bits-per-byte (BPB)** on held-out text. Primary, vocab-agnostic.
2. **Morphological competence** — higher `probe_eval.py` accuracy; `harmony` is the **pre-registered** primary sub-metric (allomorphy lives at morpheme joins).
3. **Generalization** — lower BPB and higher cloze accuracy on **rare inflected forms held out of training**.

Outcomes: (1) alone = win; (1)+(2) = the publishable result; (1)+(2)+(3) = strong. A morpheme-BPE that loses (1) but wins (2) is reported honestly as a **trade-off**, not a confirmation.

## 1. Conditions

| arm | tokenizer | vocab | role |
|---|---|---|---|
| **U32** | `sp_unigram_32000` | 32k | baseline a practitioner would ship |
| **M8** | `morpheme_bpe_8000` | ~48k | the morphology-aware contender (Pareto point) |
| U48 | `sp_unigram_48000` | 48k | ablation: same-vocab Unigram (isolates vocab-size) |
| M20 | `morpheme_bpe_20000` | ~60k | ablation: lower-fertility/less-morphology point |

Headline A/B = **U32 vs M8**. U48/M20 run only in Phase 4 to nail the vocab/embedding confound.

## 2. Architecture

Decoder-only, Llama-style (RMSNorm, SwiGLU, RoPE, no biases). RoPE keeps the context/token-count controls clean (no learned position table to mismatch).

| config | layers | d_model | heads | d_ff | non-embed params | context |
|---|--:|--:|--:|--:|--:|--:|
| **A (primary)** | 12 | 768 | 12 | 2048 | ~85M | 1024 |
| B (confirm, conditional) | 24 | 1024 | 16 | 2816 | ~300M | 2048 |

**Embedding-param control (critical).** Total params = non-embed + vocab×d_model. At d_model 768: U32 embedding ≈ 24.6M, M8 ≈ 36.9M — M8 is "bigger" by ~12M. We **hold non-embedding params identical** across arms and neutralize the embedding gap by: (a) **tied input/output embeddings** in the headline run; (b) reporting total params transparently; (c) a Phase-4 **factorized/low-rank embedding** ablation on M8 + the **U48** arm — if M8 still wins with matched effective embedding capacity, the win isn't "more params." We do **not** shrink the M8 backbone (that's a worse confound).

Headline: **Config A, context 1024, tied embeddings.**

## 3. Fair comparison — run both axes

Same text → different token counts. Each protocol has a distinct confound:

- **Match-text (PRIMARY, compute-matched).** Both arms train on the **same frozen byte stream**; they differ only in tokenization. Fix a **FLOP budget** (FLOPs ≈ 6 · params · tokens), not epochs/steps — M8 has more params but fewer tokens for the same text, so derive each arm's token budget to equalize FLOPs. Answers the science question: *given the same Turkish text, which tokenizer yields a better model?* Report the **BPB-vs-FLOPs learning curve**, not just the endpoint.
- **Match-tokens (SECONDARY).** Equal token budget (e.g. 5B each), equal steps → M8 sees ~17% more *text* (lower fertility). Flatters M8 but answers the deployment question (*at equal sequence length, which is better?*). Reporting both pre-empts "you picked the favorable budget."

**Comparing loss across vocabularies → bits-per-byte.** Per-token cross-entropy/perplexity is meaningless across tokenizers. For held-out text of `B` UTF-8 bytes with summed token NLL (nats):

```
BPB = NLL / (ln2 · B)
```

Both arms are charged the **same denominator** (raw bytes of the identical held-out text). Lower BPB = better Turkish compression regardless of tokenization (standard cross-vocab LM comparison). Also report **bits-per-character** for intuition (Turkish multi-byte letters make BPB≠BPC); BPB is primary. **Both arms must score the exact same frozen normalized surface** so the byte denominator matches — freeze one `corpus.norm.txt` and compute `B` on it.

## 4. Confounds → controls

| confound | control |
|---|---|
| vocab-size mismatch (32k vs 48k) | BPB (vocab-agnostic); U48 ablation arm |
| embedding-param gap | tied embeddings (headline); factorized-embedding ablation; report params; U48 |
| morpheme-BPE analyzer cost | one-time, cached, resumable pre-tokenization to integer IDs; **never** call the analyzer in the training loop; report as a cost line item |
| seed variance | ≥3 seeds/arm (Config A); report mean±std; require gap > 2×pooled std; paired tests |
| train/val/probe leakage | **line-level** split built *before* tokenizing; exact-dedup val vs train; report train-frequency of each probe string; build the rare-inflection set from freq-1–5 forms and **remove every train line containing them** |
| different normalized surface across arms | single frozen `corpus.norm.txt`; one byte count; commit its hash |
| tokenizer trained on val text | train both tokenizers on the **train split only** |

## 5. Metrics

All on the **same frozen held-out text / probe / rare sets** for both arms.

- **BPB (primary):** encode held-out file → IDs; teacher-forced summed NLL over strided non-overlapping windows; `BPB = ΣNLL/(ln2·bytes)`. Compare across arms (lower wins); mean±std over seeds; bootstrap CI on the gap; paired t-test across seeds. **Plot BPB vs FLOPs.**
- **Validation perplexity (secondary, within-arm only):** `exp(mean per-token NLL)`. Sanity/early-stopping. **Never compare across arms** (different token units) — state this in the report.
- **Probe accuracy (primary morphology):** `probe_eval.py` per category (agreement/possessive/harmony) + overall; random=50%, char-unigram floor≈43%. McNemar's test on the 240 paired per-item outcomes between arms. **`harmony` pre-registered** as the decisive sub-metric. Add per-byte normalization for any pair whose two members differ in byte length (cross-tokenizer fairness).
- **Rare-inflection generalization (primary):** held-out rare forms (freq 1–5, removed from train). (a) **rare-form BPB** (NLL over the rare word's bytes in context); (b) **next-morpheme cloze** — given stem+partial chain, top-1 over analyzer-generated plausible distractors.

Reports: `reports/ab_<ts>.{md,json}` (per-arm config, total/non-embed params, tokens, FLOPs, BPB mean±std+CI, within-arm PPL, probe by category + McNemar p, rare-form BPB + cloze, learning-curve data) — reusing the repo's report convention.

## 6. Implementation (from-scratch PyTorch)

New modules in `turkish-llm/`; reuse existing encoders.

- **`tokenize_corpus.py` (the keystone).** Pre-tokenize each (arm, split) **once** to a `uint16` memmap of IDs + a line-offset index (vocab<65536 for both arms).
  - U-arms: wrap the SentencePiece model (reuse the `SpmAdapter` pattern from `tr_fertility`).
  - M-arms: the expensive path. Reuse `segment_morphemes.py` (parallel analyzer ~558 w/s/8 cores) for the one-time segmentation, then `evaluate.MorphemeBPEAdapter.encode` (analyze→morphemes→merges) as the encoder.
  - **Missing piece — morpheme-BPE vocab freezer:** `morpheme_bpe.py` emits *string* pieces only. Add a freezer that enumerates base morphemes + all merged units, reserves 0–3 for `<pad>/<unk>/<s>/</s>` (match `train_tokenizer.SPECIALS`), assigns stable ints → `models/morpheme_bpe_8000.vocab.json`, with **byte-fallback** for unseen morphemes (unk-free, like the SP arms). Confirm final vocab ≈ 48k for the BPB/embedding accounting; log OOV/fallback rate on val.
  - Cache manifest = {tokenizer hash, corpus hash, normalization version}; chunked + checkpointed so a crash resumes.
- **`model.py`** — Llama-style decoder (RMSNorm/SwiGLU/RoPE/tied-embeddings), Config A/B. ~200 lines (nanoGPT-style).
- **`train_ab.py`** — minimal training loop: AMP/bf16, cosine LR + warmup, grad accumulation, `IterableDataset` over the `.ids.npy` memmap (no tokenization in the loop), FLOP-matched stop callback logging tokens/FLOPs each step, checkpoint every K steps + at the FLOP-matched and best-val-PPL points. Log seed + corpus/tokenizer hashes into each checkpoint.
- **`bpb_eval.py`** — §5 BPB over the frozen eval file (strided windows), `reports/bpb_*.{md,json}`. Reuse the report-writing pattern from `evaluate.py`/`probe_eval.py`.
- **`probe_eval.py` extension** — add a `--scorer model` branch that loads our from-scratch checkpoint + the arm's encoder directly (no HF tokenizer needed since we own `encode`/`decode`); add the per-byte fairness normalization in `evaluate()`.

## 7. Phased plan + gates (compute for Config A on one modern GPU; ÷ by GPU count)

- **Phase 0 — freeze inputs (CPU, hours).** Canonical `corpus.norm.txt`; line-level train/val split; build + excise the rare-inflection set; hash & commit; **re-train both tokenizers on the train split only**. *Gate 0:* byte counts logged; probe-string train-frequencies reported; rare forms confirmed absent from train.
- **Phase 1 — pre-tokenize + cache (CPU-bound bottleneck).** U: minutes. M: the analyzer pre-pass — at ~558 w/s/8 cores, ~1B words ≈ 21 CPU-days, so **parallelize across machines** and **this bounds corpus size**: target ~**300–500M words** (a few days, cached/resumable). *Gate 1:* both arms have hash-manifested ID memmaps; round-trip decode spot-checked; M8 final vocab ≈ 48k and cached-val fertility ≈ 1.30.
- **Phase 2 — smallest conclusive run (§8).** Config A, 1 seed, match-text+FLOP-matched, U32 vs M8. ~hours–1 GPU-day/arm. *Gate 2:* M8 BPB clearly worse and probe not better → thesis likely false at this scale, stop/pivot. Better-or-tied BPB and ≥ probe → proceed.
- **Phase 3 — headline run.** Config A, **3 seeds, both axes**. ~3–6 GPU-days/arm×seeds (parallel). *Gate 3:* BPB mean±std+CI, probe McNemar, rare-inflection metrics → win / trade-off / loss per §0.
- **Phase 4 — confirm + ablations (conditional).** Config B (≥1 seed) + factorized-embedding + U48/M20, only if Phase 3 is a win or ambiguous.

## 8. Smallest conclusive experiment

Config A (~85M non-embed), tied embeddings, context 1024, **one seed**,
**match-text + FLOP-matched**, on the Phase-1 cache (even ~100–300M words),
**U32 vs M8 only**. Metrics: BPB (primary), probe with **harmony**
pre-registered, rare-inflection BPB. **A single seed is enough to greenlight
or kill**; multi-seed Phase 3 is required only to publish a positive claim.

## 9. Risks

- Embedding capacity, not morphology, drives a win → tied embeddings + factorized ablation + U48.
- Match-protocol flips the result → report both; lead with match-text; never report only the favorable one.
- Probe too easy / leaked → rare-inflection set is the decisive, leakage-controlled morphology test; probe is supporting.
- Analyzer cost caps corpus → undertrained regime hides effects → size corpus to the cache budget; report "at this data scale."
- M8 byte-fallback inflates token counts on rare forms → contaminates BPB → log fallback rate; fix the freezer before trusting BPB.

## 10. Open parameters to set at Phase 0/1

- Final experiment corpus size (bound by the analyzer pre-pass budget across your machines).
- FLOP budget / token budgets per arm (derive from the chosen corpus size and a Chinchilla-ish ~20 tokens/param starting point).
- Seed count for Phase 3 (≥3).
- GPU allocation (how many of the machines for parallel seeds/arms).
