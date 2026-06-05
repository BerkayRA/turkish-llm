# turkish-llm

Tooling and experiments toward a **Turkish language model** — starting with
the tokenizer. It consumes the morphological analyzer
[`turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer) (a git
submodule) to prepare and clean training data and, crucially, to **measure and
improve how well a subword tokenizer respects Turkish morphology**.

> This is the LLM/training side of the project, kept separate from the clean,
> dependency-free `turkish-tokenizer` library. Tokenizer training is CPU-only;
> the GPUs come in later, for the model itself.

## Why a custom tokenizer

Turkish is agglutinative, so English/multilingual subword tokenizers fragment
it badly — high **fertility** (tokens per word) wastes compute, shortens
effective context, and hurts modeling. Training a Turkish-specific vocabulary
fixes the bulk of that; using the morphological analyzer lets us push further
and align tokenization with morpheme boundaries.

## Setup

```bash
git clone --recurse-submodules <this repo>
# or, after a plain clone:
git submodule update --init --recursive

python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Build corpora (normalized surface text + morpheme-segmented text).
#    Defaults to the UD smoke corpus; pass --text for a real corpus.
python prepare_corpus.py --out data

# 2. Train candidate tokenizers (vocab sizes are free — use 16k–64k for real).
python train_tokenizer.py --vocab-sizes 16000 32000 48000 --byte-bpe 32000

# 3. Compare them on Turkish morphological fitness.
#    Writes a detailed report (reports/eval_*.md + .json) by default.
python evaluate.py --corpus data/corpus.raw.txt
```

Every evaluation writes a version-controlled report to `reports/` — run
metadata (corpus word count, vocab sizes, algorithms), the full metric table,
and auto-generated commentary on relative performance. See `docs/EVALUATION.md`.

Three tokenizer families are trained and compared:
- **`sp_unigram_<V>`** — SentencePiece Unigram on raw text (the standard baseline).
- **`byte_bpe_<V>`** — byte-level BPE (GPT/Llama-style).
- **`sp_morph_<V>`** — SentencePiece trained on **morpheme-segmented** text, so
  its merges are biased to respect morpheme boundaries (morphology-informed vocab).

plus a **`morpheme-aware`** tokenizer evaluated at inference: segment each word
into morphemes with the analyzer, then subword-encode each morpheme
separately — pieces never cross a morpheme boundary (boundary recall = 1.0 by
construction).

`evaluate.py` reports, via the bundled `tr_fertility`, **fertility**,
tokens/morpheme, single-token-word %, and **morpheme-boundary precision/recall/F1**.

## Smoke-test results (illustrative only)

Trained and evaluated on the tiny UD-Turkish-IMST corpus (~46k words). Absolute
numbers are inflated by the small corpus + small vocab; a real corpus with a
32k+ vocab yields much lower fertility. The **relative** picture is the point:

| tokenizer | fertility | tok/morph | single% | bound-F1 | recall |
|---|---:|---:|---:|---:|---:|
| sp_unigram_8000 | **1.66** | 0.82 | 58% | 0.33 | 0.27 |
| sp_unigram_4000 | 1.96 | 0.98 | 48% | 0.41 | 0.39 |
| sp_morph_8000 | 2.30 | 1.14 | 28% | 0.65 | 0.72 |
| byte_bpe_4000 | 2.68 | 1.33 | 13% | 0.33 | 0.48 |
| morpheme-aware | 3.11 | 1.54 | 29% | **0.83** | **1.00** |
| char (baseline) | 6.45 | 3.21 | 1% | 0.31 | 1.00 |

**Takeaway — a fertility ↔ morphology tradeoff.** Plain Unigram at higher vocab
minimizes fertility but its cuts largely ignore morphemes (recall 0.27).
Training on morpheme-segmented text raises morpheme recall to ~0.72 at a
fertility cost; the segment-then-subword tokenizer guarantees morpheme
alignment (recall ~1.0) at the highest token cost. Which point on this curve is
best is an empirical question for *downstream model quality* — and this repo is
set up to answer it.

## Roadmap

1. **Tokenizer** (here): real corpus → vocab-size sweep (16k–64k) → pick by
   fertility *and*, eventually, downstream loss. Decide vanilla vs
   morphology-aware.
2. **Model pre-training**: the heavy stage, on the GPU machines. Architecture
   and size are open (the hardware supports multi-billion-parameter models).
3. **Inference/deployment**: including edge targets (Jetson) via quantization.
4. **Learning project**: reimplement BPE (and Unigram) from scratch to compare
   against the library tokenizers and the morphology-aware variants (see the
   `turkish-tokenizer` SUGGESTIONS).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — architecture, the two-repo/submodule split, the tokenizer families
- [docs/METRICS.md](docs/METRICS.md) — fertility, tokens/morpheme, boundary precision/recall/F1, how to read them
- [docs/USAGE.md](docs/USAGE.md) — setup and commands (prepare / train / evaluate)
- [docs/EVALUATION.md](docs/EVALUATION.md) — methodology, reports, choosing a tokenizer
- [docs/PROBING.md](docs/PROBING.md) — morphological probing benchmark (agreement, possessive, harmony)

## Layout

```
prepare_corpus.py     UD/text -> normalized + morpheme-segmented corpora
train_tokenizer.py    SentencePiece Unigram / byte-level BPE / morpheme-aware
evaluate.py           fertility + morpheme-boundary comparison table
_tok.py               makes the turkish-tokenizer submodule importable
vendor/turkish-tokenizer/   the morphological analyzer (submodule)
```
