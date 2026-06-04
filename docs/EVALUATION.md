# Evaluation

## Methodology

`evaluate.py` scores each candidate tokenizer over a corpus with
`tr_fertility.score_corpus`, using the morphological analyzer as the reference
for morpheme boundaries. It always includes two baselines and a morpheme-aware
tokenizer so every run is self-anchoring:

- **`whitespace`** — fertility floor (1.0), no morphology. Sanity anchor.
- **`char`** — fertility ceiling, boundary recall ~1.0 (cuts everywhere),
  precision low. The "maximally over-segmented" anchor.
- **`morpheme-aware`** — segment-then-subword over the largest plain Unigram;
  boundary recall ~1.0 by construction. The morphology-aligned anchor.

A real tokenizer should sit *between* whitespace and char on fertility, and the
interesting question is how much morpheme alignment it achieves at a given
fertility.

## Reports (default behaviour)

After every evaluation (unless `--no-report`), a detailed report is written:

- `reports/eval_<UTC-timestamp>.md` — human-readable: run metadata, a results
  table, auto-generated commentary on relative performance, and metric
  definitions.
- `reports/eval_<UTC-timestamp>.json` — the same data, machine-readable, for
  plotting / regression tracking across runs.
- `reports/latest.md` — copy of the most recent markdown report.

Each report records: **timestamp, corpus path, total corpus word count, words
actually scored (and the `--limit`), morphemes/word of the corpus, every
tokenizer's algorithm and vocab size**, and the full metric set
(fertility, tokens/morpheme, single-token %, boundary precision/recall/F1).
Reports are committed to the repo as a version-controlled audit trail.

The commentary is generated programmatically: it flags the lowest-fertility
tokenizer, the best morpheme alignment, the vocab-size effect within the
Unigram family, and the fertility cost of the morpheme-aware approaches.

## How to choose a tokenizer

1. **Filter by fertility** on the real corpus — eliminate anything with
   fertility much above the Unigram baseline.
2. **Among comparable-fertility candidates, prefer higher boundary F1/recall**
   — morphological alignment tends to help a model generalize across the
   inflectional paradigm.
3. **Confirm with downstream loss.** Fertility and alignment are cheap proxies;
   the deciding evidence is the model's training/validation loss with each
   tokenizer. Pick 2–3 finalists here, then compare on a small model run.

## Smoke-test snapshot (UD-Turkish-IMST, ~46k words)

Illustrative only — tiny corpus + small vocab inflate absolute fertility; the
*ranking* and the *shape of the tradeoff* are the takeaways.

| tokenizer | fertility | bound-F1 | recall |
|---|---:|---:|---:|
| sp_unigram_8000 | **1.66** | 0.33 | 0.27 |
| sp_morph_8000   | 2.30 | 0.65 | 0.72 |
| byte_bpe_4000   | 2.68 | 0.33 | 0.48 |
| morpheme-aware  | 3.11 | **0.83** | **1.00** |
| char (baseline) | 6.45 | 0.31 | 1.00 |

**Reading it:** plain Unigram at the largest vocab minimizes fertility but its
cuts mostly ignore morphemes (recall 0.27). Training on morpheme-segmented text
roughly triples recall (0.72) for a ~40% fertility increase. The
segment-then-subword tokenizer guarantees alignment (recall ~1.0) at the
highest token cost. Which point wins is an empirical, downstream question — but
the curve is exactly what we wanted to be able to see.

## Expectations on the real corpus

With a large corpus and a 32k+ vocab, expect:
- Plain Unigram fertility to drop substantially (toward ~1.5 or lower).
- Single-token-word % to rise (more frequent words become atomic).
- The *relative* ordering and the fertility ↔ alignment tradeoff to persist.

Re-run `evaluate.py` after training on the real corpus; compare the new
`reports/*.json` against the smoke run to quantify the gain.
