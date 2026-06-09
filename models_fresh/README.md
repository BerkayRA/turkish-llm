# models_fresh — tokenizers retrained on fresh, held-out data (2026-06-09)

These tokenizers were **retrained from scratch on a fresh corpus** to check that the earlier
fertility advantage of the morpheme-aware tokenizer was real and **not an artifact of
overfitting / train–eval overlap**. They are evaluated on a **disjoint, held-out** eval set
(zero document overlap with training), so the numbers below are honest out-of-sample fertility.

> Committed deliberately despite the repo's `.gitignore` (`*.model`/`*.vocab`/`models/` are
> normally excluded): these are small (~5 MB total) and are the *evidence* for the anti-overfit
> result. `git add -f` was used for the SentencePiece binaries.

## Corpus provenance

Built with the sibling **`turkish-corpus`** pipeline (HPLT v2 `tur_Latn` + Turkish Wikipedia,
cleaned: Turkish-aware normalization, Wikipedia-calibrated Gopher/quality filters, MinHash
dedup), then split into disjoint sets with `scripts/export_text_corpus.py`:

| Split | Docs | Size | Notes |
|-------|-----:|-----:|-------|
| train | 167,503 | 442 MB | sp_unigram + byte_bpe trained on a 100k-line (~266 MB) slice; morpheme_bpe on a 5,000-line `segment_morphemes.py` sample |
| eval (held-out) | 8,816 | 23 MB | **0 exact overlap with train**; `evaluate.py --limit 8000` |

Persisted corpus: `~/corpora/turkish-fresh-20260609/` (`train.txt`, `eval.txt`,
`sample.morph.fresh.txt`, `cleaned_jsonl/`).

## Held-out fertility (lower = fewer tokens/word = better)

| Tokenizer | Fertility | tok/morph | single% | boundary-F1 | recall |
|-----------|----------:|----------:|--------:|------------:|-------:|
| whitespace (baseline) | 1.000 | 0.523 | 100.0% | 0.000 | 0.000 |
| **morpheme_bpe_20000** | **1.158** | 0.606 | 85.0% | 0.270 | 0.156 |
| morpheme_bpe_8000 | 1.232 | 0.645 | 78.1% | 0.379 | 0.234 |
| sp_unigram_32000 | 1.465 | 0.767 | 72.2% | 0.248 | 0.178 |
| sp_unigram_16000 | 1.609 | 0.842 | 63.6% | 0.315 | 0.248 |
| sp_morph_32000 | 2.039 | 1.067 | 35.1% | **0.657** | **0.643** |
| sp_morph_16000 | 2.073 | 1.085 | 35.0% | 0.649 | 0.647 |
| byte_bpe_32000 | 2.198 | 1.150 | 24.4% | 0.274 | 0.357 |
| morpheme-aware (analyzer only) | 2.734 | 1.431 | 34.5% | 0.879 | 0.999 |
| char | 6.504 | 3.404 | 1.4% | 0.283 | 0.999 |

Full reports: `reports/eval_20260609T102433Z.{md,json}` (without sp_morph) and
`reports/eval_20260609T125623Z.{md,json}` (with sp_morph).

## Key findings

- **The morpheme-aware advantage is real, not overfit.** `morpheme_bpe_20000` leads at
  **1.158 tokens/word** on unseen text — *better* than its prior in-distribution 1.22 — while
  training on far less data (a 5k-line morph sample).
- **`byte_bpe_32000` degraded out-of-sample (1.78 → 2.20)** — it had overfit its training
  distribution more than the morphology-aware models, which generalize better.
- **`sp_morph` trades fertility for alignment.** Trained on morpheme-segmented text but
  tokenizing raw text at inference (no analyzer needed → fast, HF-exportable, *deployable*),
  it is far more morpheme-aligned than any other subword model (**boundary-F1 0.657, recall
  0.643** vs sp_unigram's 0.25/0.18) — but its fertility (**2.039**) is *worse* than
  sp_unigram (1.465). So the fast-deployable morpheme-aware tokenizer does **not** match
  `morpheme_bpe`'s compression; it's an alignment↔compression tradeoff.
  *Caveat:* sp_morph was trained on the small 5k-line morph sample and SentencePiece
  saturated at ~19k pieces (16k≈32k results confirm vocab starvation) — more
  morpheme-segmented training data would likely lower its fertility.
- **Practical takeaway:** best overall = `morpheme_bpe_20000` (1.158) but needs `tr_api` at
  inference; best *deployable for compression* = `sp_unigram_32000` (1.465); best *deployable
  for morpheme alignment* = `sp_morph_32000` (boundary-F1 0.66, fertility 2.04).

## Reproduce

```bash
# 1. Fresh corpus + disjoint split (in the turkish-corpus repo):
uv run python scripts/export_text_corpus.py \
    --clean-dir <cleaned_hplt>/final --clean-dir <cleaned_wiki>/final \
    --out-dir <out> --eval-fraction 0.05

# 2. Train (this repo, .venv):
.venv/bin/python train_tokenizer.py --raw <out>/train.txt --out models_fresh \
    --vocab-sizes 16000 32000 --byte-bpe 32000 --no-morph --input-sentence-size 2000000
.venv/bin/python segment_morphemes.py <out>/train.txt -o sample.morph.fresh.txt \
    --max-lines 5000 --workers 8
.venv/bin/python morpheme_bpe.py train sample.morph.fresh.txt --merges 20000 \
    -o models_fresh/morpheme_bpe_20000.json   # and --merges 8000

# 3. Evaluate on the HELD-OUT eval (not training data):
.venv/bin/python evaluate.py --corpus <out>/eval.txt --models models_fresh --limit 8000
```

## Files

- `byte_bpe_32000.json` — byte-level BPE (HF `tokenizers` format; loadable anywhere).
- `morpheme_bpe_8000.json`, `morpheme_bpe_20000.json` — morpheme-aware BPE (custom: analyzer
  segmentation + merges; decode via `morpheme_bpe.MorphemeBPE.from_file`).
- `sp_unigram_16000.{model,vocab}`, `sp_unigram_32000.{model,vocab}` — SentencePiece Unigram baselines (trained on raw surface text).
- `sp_morph_16000.{model,vocab}`, `sp_morph_32000.{model,vocab}` — SentencePiece Unigram trained on
  **morpheme-segmented** text (deployable morpheme-aware tokenizer: no analyzer at inference).
  Trained via `train_tokenizer.train_sentencepiece(sample.morph.fresh.txt, "models_fresh/sp_morph_<V>", V)`.
