# Metrics

All metrics compare a **subword tokenizer** against the **morphological
analyzer** (the reference for "true" morpheme boundaries), micro-averaged over
the word tokens of a corpus. They are computed by
`tr_fertility.score_corpus` in the `turkish-tokenizer` submodule.

For each word: the subword tokenizer produces *pieces*; the analyzer produces
*morphemes*. Example for `kitabımı` ("my book", accusative):

```
subword (sp_unigram): ["▁kitab", "ımı"]        -> 2 pieces, cut at offset 5
morphemes (analyzer): kitab | ım | ı           -> 3 morphemes, cuts at 5, 7
```

## Fertility — subword tokens per word
```
fertility = total subword tokens / total words
```
**The headline number. Lower is better.** Each extra token is paid on every
training and inference step, and eats into the context window. For Turkish,
English-centric tokenizers run high (2–3+); a Turkish-trained vocab is much
lower. A whitespace "tokenizer" has fertility 1.0 (one token per word) — the
unreachable floor; `char` is the worst case (one token per character).

## Tokens per morpheme
```
tok/morph = total subword tokens / total morphemes
```
Fertility normalized by morphological complexity. ~1.0 means the tokenizer
emits roughly one piece per morpheme; <1.0 means pieces span multiple
morphemes (good compression but morphologically opaque); >1.0 means it splits
*within* morphemes.

## Morphemes per word
```
morphemes/word = total morphemes / total words
```
Not a tokenizer metric — it characterizes the **corpus** (how agglutinative
the text is) and anchors the other two. Turkish typically ~2–3.

## Single-token-word %
Fraction of words emitted as exactly one subword token. Higher means more
words are atomic in the vocabulary (good for frequent words; impossible for
the long tail of inflected forms).

## Boundary alignment — precision / recall / F1
Do the tokenizer's internal cut points fall on **true morpheme boundaries**?
For each word we take the set of internal char offsets where the subword pieces
cut, and the set where morphemes cut, then:

```
precision = |subword cuts ∩ morpheme cuts| / |subword cuts|
recall    = |subword cuts ∩ morpheme cuts| / |morpheme cuts|
F1        = harmonic mean
```

- **Recall** — what fraction of real morpheme boundaries the tokenizer
  captured. High recall = the tokenizer "sees" the morphology. The
  morpheme-aware tokenizer is ~1.0 by construction; `char` is ~1.0 trivially
  (it cuts everywhere).
- **Precision** — what fraction of the tokenizer's cuts are real morpheme
  boundaries. `char` has low precision (most char cuts aren't morpheme
  boundaries); a morpheme-aligned tokenizer has high precision.
- **F1** balances the two. It is the single best "is this tokenizer
  morphologically sensible?" score — but read it *alongside* fertility, since
  you can trivially max recall by over-segmenting.

### "% reconstructible" caveat
Boundary alignment needs the pieces to reconstruct the word (after stripping
metaspace markers like `▁`, `Ġ`, `##`). Byte-level BPE can map to raw bytes
that don't cleanly reconstruct a Unicode word; those words are skipped for the
boundary metric (but still counted for fertility). The report shows what
fraction was reconstructible.

## How to read them together

There is a **fertility ↔ morphology tradeoff**:
- Push vocab up → fertility drops, but cuts increasingly ignore morphemes
  (recall falls).
- Force morpheme alignment (train on segmented text, or segment-then-subword)
  → recall rises toward 1.0, but fertility rises too.

The "best" point depends on **downstream model quality**, which fertility
alone can't tell you. These metrics narrow the field cheaply (CPU, minutes)
before you spend GPU hours; the final call is made by training loss /
evaluation on the actual model.

## Caveats
- On a **tiny corpus** (e.g. the UD smoke set) absolute fertility is inflated
  and vocab sizes are capped — trust the *relative* ranking, not the absolute
  numbers. Re-measure on the real corpus.
- Metrics are computed over **word tokens** (the analyzer's notion of a word),
  lowercased; punctuation and whitespace are excluded.
