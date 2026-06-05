# Morphological Probing Report

- **Generated:** 20260605T074742Z
- **Scorer:** char_unigram(corpus.train.txt)
- **Dataset:** `benchmarks/probe.jsonl` (240 minimal pairs)
- **Overall accuracy:** 43.3% (random = 50%)

## By category

| category | pairs | accuracy |
|---|---:|---:|
| agreement | 80 | 50.0% |
| harmony | 80 | 55.0% |
| possessive | 80 | 25.0% |

## Notes

- Accuracy = fraction of pairs where the model assigns higher probability to the grammatical sentence. 50% = chance.
- A char-unigram baseline near 50% confirms the pairs require real morphological knowledge, not surface statistics.
