# Usage

## Setup

```bash
git clone --recurse-submodules https://github.com/BerkayRA/turkish-llm
cd turkish-llm
# if you cloned without --recurse-submodules:
git submodule update --init --recursive

python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # sentencepiece, tokenizers (CPU-only)
```

The morphological analyzer is the pinned submodule at
`vendor/turkish-tokenizer`; `_tok.py` puts it on `sys.path`, so the scripts can
`import tr_api`, `tr_normalize`, `tr_fertility`.

## 1. Prepare corpora

```bash
# Smoke test: defaults to the UD treebank bundled with the tokenizer checkout.
python prepare_corpus.py --out data

# Real corpus: one document/sentence per line.
python prepare_corpus.py --text /path/to/turkish_corpus.txt --out data
```

Produces `data/corpus.raw.txt` (normalized surface) and
`data/corpus.morph.txt` (morpheme-segmented). Both are line-aligned. Progress
and word/OOV counts go to stderr.

## 2. Train tokenizers

```bash
# Smoke (tiny corpus): small vocab sizes only.
python train_tokenizer.py --vocab-sizes 2000 4000 8000 --byte-bpe 4000

# Real run: full vocab sweep.
python train_tokenizer.py --vocab-sizes 16000 32000 48000 64000 --byte-bpe 32000
```

Flags:
- `--vocab-sizes N [N ...]` — trains `sp_unigram_<N>` and `sp_morph_<N>` per size.
- `--byte-bpe N` — byte-level BPE vocab size (`0` to skip).
- `--no-morph` — skip the morpheme-segmented variant.
- `--raw` / `--morph` — corpus paths (default `data/corpus.{raw,morph}.txt`).
- `--out` — model dir (default `models/`).

Models are written to `models/` (gitignored): `sp_*.model`/`.vocab`,
`byte_bpe_*.json`.

## 3. Evaluate

```bash
python evaluate.py --corpus data/corpus.raw.txt --limit 8000
```

Flags:
- `--corpus` — text to score (default `data/corpus.raw.txt`).
- `--limit N` — cap word tokens scored (the analyzer is the per-word cost;
  raise/remove for a full pass on the real corpus).
- `--models` — model dir.
- `--report-dir` — where reports go (default `reports/`).
- `--no-report` — skip the report (it is written by default).

Prints a comparison table to stdout and, **by default after every run**,
writes a detailed report to `reports/eval_<timestamp>.{md,json}` plus
`reports/latest.md`. See `EVALUATION.md`.

## Scoring an external tokenizer directly

You can score any tokenizer against the morphology without this repo's trainer,
straight from the submodule:

```bash
python vendor/turkish-tokenizer/tr_fertility.py --spm your.model corpus.txt
python vendor/turkish-tokenizer/tr_fertility.py --tokenizers-json your.json corpus.txt
python vendor/turkish-tokenizer/tr_fertility.py --hf <name-or-dir> corpus.txt
python vendor/turkish-tokenizer/tr_fertility.py --tokenizer char corpus.txt   # baseline
```

## Hardware

Everything here is **CPU-only** — tokenizer training (SentencePiece/BPE) does
not use the GPU. The GPUs are for the model pre-training stage (later).
