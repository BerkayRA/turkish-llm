# Status — where we are

_Snapshot for picking back up. Last updated end of 2026-06-05._

## The project in one line
Building a Turkish LLM whose differentiator is a **morphology-aware tokenizer**.
Two repos: [`turkish-tokenizer`](https://github.com/BerkayRA/turkish-tokenizer)
(the dependency-free morphological analyzer, a git submodule here) and this
`turkish-llm` (tokenizer training, evaluation, and the upcoming model A/B).

## What's done ✅
- **Tokenizers trained** on the cleaned real corpus (491M-word web export →
  cleaned to 36.9M lines via `clean_corpus.py`): SentencePiece-Unigram 16k/32k/48k,
  byte-BPE 32k, and **morpheme-BPE** (`morpheme_bpe.py`, from-scratch BPE over
  morphemes). Models in `models/` (gitignored).
- **Tokenizer-level result (the UVP):** morpheme-BPE **Pareto-dominates** Unigram
  — `morpheme_bpe_8000` (~48k vocab) fertility **1.30** + morpheme recall **0.27**
  vs `sp_unigram_32000` 1.53 + 0.16 (lower fertility *and* higher alignment).
  See `reports/latest.md`, `docs/EVALUATION.md`.
- **Evaluation harnesses:** `evaluate.py` (fertility / boundary alignment, auto-report),
  `probe_eval.py` + `benchmarks/probe.jsonl` (240-pair morphological probing
  benchmark; char-unigram baseline ~43% ≈ chance validates it). Docs:
  `docs/{DESIGN,METRICS,USAGE,EVALUATION,PROBING}.md`.
- **A/B experiment fully spec'd:** `docs/EXPERIMENT_AB.md` — U32 vs M8, ~125M
  Llama-style decoder, from-scratch PyTorch, both fairness axes, bits-per-byte
  primary metric, confound controls, phased plan + decision gates, concrete
  hyperparameters/budget/artifact layout.
- **Cluster ops guide:** `docs/GPU_OPS.md` (NVIDIA BCM + Slurm + torchrun + rsync staging).

## What's next ▶ (Phase 0–1, all CPU — no GPU yet)
1. **Freeze inputs:** build canonical `corpus.norm.txt`, line-level train/val/test
   split, the rare-inflection held-out set; re-train both tokenizers on the train
   split only; commit hashes. (`docs/EXPERIMENT_AB.md` §11.3)
2. **Morpheme-BPE vocab freezer** — the one missing build piece: `morpheme_bpe.py`
   emits *string* pieces only; add a string→int vocab (specials + 256 byte-fallback
   + morphemes + merges) → `models/morpheme_bpe_8000.vocab.json`. (§11.4)
3. **`tokenize_corpus.py`** — cache both arms to `uint16` `.ids.npy` memmaps
   (analyzer pre-pass for M8 is the CPU bottleneck; shard across the cluster, §6 of GPU_OPS).
Then Phase 2 (GPU): `model.py` + `train_ab.py` + `bpb_eval.py` + `probe_eval --scorer model`
→ the smallest single-seed run that greenlights or kills the thesis.

## Open decisions to make next session
- Final experiment **corpus size** (bounded by the analyzer pre-pass budget across the fleet).
- **FLOP/token budgets** per arm; **seed count** for Phase 3 (≥3).
- **GPU/CPU allocation** on the cluster (drives corpus size + parallelism).

## Gotchas / must-remember 🚩
- The 3.4 GB corpus (`ts_corpus_ver_2-export.txt`) is **gitignored** — **never `git add -A`** with it present (it once got committed; push rejected; cleaned up).
- The morphological analyzer is **~114 words/s** single-thread (~558/s on 8 cores). It bounds anything that segments the full corpus → cache once, parallelize, never call it in a training loop.
- morpheme-BPE **needs the analyzer to encode** (decode is free) → its lower fertility is real but comes with a preprocessing cost.
- Across tokenizers, compare **bits-per-byte**, never per-token loss/perplexity.
- `turkish-tokenizer` stays **dependency-free**; heavy deps (torch, etc.) live only here.

## Pointers
- Spec: `docs/EXPERIMENT_AB.md` · Cluster: `docs/GPU_OPS.md` · Metrics: `docs/METRICS.md`
- Latest tokenizer eval: `reports/latest.md` · Probing: `reports/probe_latest.md`
