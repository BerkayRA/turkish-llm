# Tokenizer Evaluation Report

- **Generated:** 20260604T111131Z
- **Corpus:** `data/corpus.raw.txt` — 46,326 words total, 8,000 scored (limit 8000)
- **Morphological reference:** turkish-tokenizer (2.013 morphemes/word on the scored slice)
- **Tokenizers compared:** 10

## Results (sorted by fertility, low → high)

| tokenizer | algorithm | vocab | fertility | tok/morph | single% | bound-P | bound-R | bound-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `whitespace` | whitespace (baseline) | — | 1.000 | 0.497 | 100.0% | 0.000 | 0.000 | 0.000 |
| `sp_unigram_8000` | SentencePiece Unigram | 8000 | 1.659 | 0.824 | 58.0% | 0.435 | 0.265 | 0.329 |
| `sp_unigram_4000` | SentencePiece Unigram | 4000 | 1.962 | 0.975 | 47.7% | 0.438 | 0.388 | 0.411 |
| `sp_morph_8000` | SentencePiece Unigram, morpheme-trained | 8000 | 2.303 | 1.144 | 27.9% | 0.590 | 0.717 | 0.647 |
| `sp_morph_4000` | SentencePiece Unigram, morpheme-trained | 4000 | 2.341 | 1.163 | 27.8% | 0.574 | 0.720 | 0.639 |
| `sp_unigram_2000` | SentencePiece Unigram | 2000 | 2.371 | 1.178 | 36.9% | 0.404 | 0.512 | 0.451 |
| `byte_bpe_4000` | byte-level BPE | 4000 | 2.681 | 1.332 | 12.8% | 0.249 | 0.480 | 0.328 |
| `sp_morph_2000` | SentencePiece Unigram, morpheme-trained | 2000 | 2.683 | 1.333 | 25.6% | 0.494 | 0.786 | 0.607 |
| `morpheme-aware` | morpheme-aware (segment-then-subword) | — | 3.108 | 1.544 | 29.3% | 0.712 | 0.999 | 0.831 |
| `char` | char (baseline) | — | 6.451 | 3.205 | 1.0% | 0.185 | 0.999 | 0.312 |

## Commentary

- **Lowest fertility (best compression):** `sp_unigram_8000` at 1.659 tokens/word.
- **Best morpheme alignment (boundary F1):** `morpheme-aware` at F1 0.831 (recall 0.999 for `char`).
- **Vocab-size effect:** raising the SentencePiece Unigram vocab 2000→8000 moved fertility 2.371→1.659 (down).
- **Morpheme-awareness tradeoff:** the segment-then-subword tokenizer reaches boundary recall 0.999 (vs 0.265 for `sp_unigram_8000`) at fertility 3.108 (vs 1.659) — more tokens, but morpheme-aligned by construction.
- **Training on morpheme-segmented text** (`sp_morph_8000`) lifts boundary recall to 0.717 (vs 0.265 for the raw-trained `sp_unigram_8000`) at fertility 2.303 vs 1.659.

## Metric definitions

- **fertility** — subword tokens per word; lower is better (less compute, longer effective context).
- **tok/morph** — subword tokens per morpheme (analyzer reference).
- **single%** — words emitted as exactly one token.
- **boundary P/R/F1** — do subword cut points fall on true morpheme boundaries? Recall = fraction of morpheme boundaries captured; precision = fraction of subword cuts that are morpheme boundaries.

See `docs/METRICS.md` for full definitions and interpretation.

