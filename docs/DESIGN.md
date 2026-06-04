# Design

This repo is the **training / LLM side** of the Turkish-LLM project. It is
kept separate from [`turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer)
— the morphological analyzer — which stays a clean, dependency-free library.

## Why two repos

`turkish-tokenizer` is a reusable, stdlib-only library: a morphological
analyzer. The LLM side needs heavy, churn-prone dependencies (`sentencepiece`,
`tokenizers`, later `torch`, `datasets`, …) on its own release cadence.
Mixing them would pollute the library's dependency profile and couple two very
different lifecycles. So:

- `turkish-tokenizer` — the analyzer. No third-party deps.
- `turkish-llm` (here) — consumes it as a **git submodule** (`vendor/turkish-tokenizer`),
  imported via `_tok.py`, which puts the submodule on `sys.path`.

A consumer clones with `--recurse-submodules`; the submodule is pinned to a
specific tokenizer commit, so results are reproducible.

## Why a custom tokenizer at all

Turkish is agglutinative: one stem carries many suffixes, so the surface
vocabulary is effectively unbounded. English/multilingual subword tokenizers
fragment Turkish into many pieces per word — high **fertility** — which wastes
compute, shortens effective context, and makes modeling harder. A
Turkish-trained vocabulary fixes most of that; the analyzer lets us go further
and *measure/encourage* alignment with morpheme boundaries.

The analyzer is **not** the model's tokenizer. It is used to (1) clean and
normalize training data, (2) evaluate candidate subword tokenizers, and (3)
build morphology-aware tokenizers.

## Pipeline

```
prepare_corpus.py   raw text / UD  ─▶  data/corpus.raw.txt   (normalized surface)
                                   └▶  data/corpus.morph.txt (morpheme-segmented)
train_tokenizer.py  corpora        ─▶  models/*.model | *.json   (candidate tokenizers)
evaluate.py         models+corpus  ─▶  stdout table + reports/eval_*.{md,json}
```

### `prepare_corpus.py`
Extracts text (UD `# text =` sentences or `--text` lines), runs it through the
analyzer's normalizer (`tr_normalize`) to produce two parallel corpora:
- **`corpus.raw.txt`** — normalized surface text (attached clitics split). What
  a normal subword tokenizer trains on.
- **`corpus.morph.txt`** — morpheme-segmented text, each morpheme a whitespace
  token. Training on this biases a subword model toward morpheme-aligned pieces.

### `train_tokenizer.py`
Trains three families so they can be compared:
- **`sp_unigram_<V>`** — SentencePiece Unigram on raw text. The standard
  baseline. Unigram is a probabilistic model (start large, prune to maximize
  likelihood) that tends to be kinder to morphology than greedy BPE.
- **`byte_bpe_<V>`** — byte-level BPE (Hugging Face `tokenizers`), GPT/Llama
  style; operates on bytes so it never emits `<unk>`.
- **`sp_morph_<V>`** — SentencePiece Unigram trained on `corpus.morph.txt`. A
  *morphology-informed vocabulary*: merges rarely cross morpheme boundaries.

Common settings: `byte_fallback=True` (unk-free), `character_coverage=0.9995`,
NFKC normalization, special tokens `<pad>/<unk>/<s>/</s>`. Vocab sizes are a
free parameter (`--vocab-sizes`); small defaults only suit the tiny smoke
corpus — real runs use 16k–64k.

### `evaluate.py`
Scores every trained tokenizer (plus `whitespace`/`char` baselines and a
**morpheme-aware** tokenizer) against the analyzer using
`tr_fertility.score_corpus`. Prints a comparison table and, by default, writes
a detailed report to `reports/` (see `EVALUATION.md`).

The **morpheme-aware** tokenizer is the strongest way to leverage the analyzer:
segment each word into morphemes, subword-encode each morpheme separately, and
concatenate. Pieces never cross a morpheme boundary, so boundary recall is
~1.0 by construction — at a fertility cost. It quantifies the ceiling of the
fertility ↔ morphology tradeoff.

## How this feeds the LLM

The tokenizer is chosen by fertility (and, eventually, downstream model loss),
then frozen. Model pre-training — the heavy, GPU stage — comes next. Tokenizer
training itself is CPU-only.

See `METRICS.md` for what the numbers mean, `USAGE.md` for commands, and
`EVALUATION.md` for methodology and how to read reports.
