"""
morpheme_bpe.py — a Byte-Pair-Encoding tokenizer whose base units are
MORPHEMES, not characters/bytes.

This is the "soft, morphology-informed" tokenizer. Instead of forbidding
merges across morpheme boundaries (the rigid segment-then-subword approach,
floored at ~morphemes/word ≈ 2.0 fertility), it treats morphemes as the
alphabet and learns BPE merges of frequent adjacent morphemes (within a
word). So:

  - every token is a whole sequence of morphemes → token boundaries are
    ALWAYS morpheme boundaries (boundary precision = 1.0 by construction);
  - frequent morpheme sequences (e.g. -iyor-um, -lar-ı-nda) collapse into
    single tokens → fertility drops below the rigid floor, toward Unigram;
  - the number of merges trades fertility against boundary recall.

It is implemented from scratch (the classic incremental BPE with a
pair→words index), which also serves as the BPE-from-scratch exercise.

Training input: morpheme-segmented text from segment_morphemes.py, where
words are space-separated and morphemes within a word are joined by `▁`.

Encoding a word at inference: analyse it into morphemes (the morphological
analyzer), then apply the learned merges. Decoding needs no analyzer —
tokens concatenate back to the surface.

Usage:
    python morpheme_bpe.py train data/sample.morph.txt --merges 8000 \
        -o models/morpheme_bpe_8000.json
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

MORPH_SEP = "▁"


# -----------------------------------------------------------------------------
# Training (incremental BPE over morphemes)
# -----------------------------------------------------------------------------

def read_word_freqs(path):
    """Word frequencies as {(morpheme, ...): count} from a segmented corpus."""
    freqs = collections.Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                morphs = tuple(word.split(MORPH_SEP))
                freqs[morphs] += 1
    return freqs


def learn_bpe(word_freqs, num_merges, min_freq=2):
    """Learn up to `num_merges` morpheme merges. Returns an ordered list of
    (a, b) pairs. Incremental: only words containing the merged pair are
    touched per step."""
    words = [list(w) for w in word_freqs]
    freqs = [word_freqs[w] for w in word_freqs]

    pair_freq = collections.Counter()
    pair_where = collections.defaultdict(set)   # pair -> set of word indices

    def add_word_pairs(i):
        sym, f = words[i], freqs[i]
        for k in range(len(sym) - 1):
            p = (sym[k], sym[k + 1])
            pair_freq[p] += f
            pair_where[p].add(i)

    def remove_word_pairs(i):
        sym, f = words[i], freqs[i]
        for k in range(len(sym) - 1):
            p = (sym[k], sym[k + 1])
            pair_freq[p] -= f
            if pair_freq[p] <= 0:
                del pair_freq[p]
            pair_where[p].discard(i)

    for i in range(len(words)):
        add_word_pairs(i)

    merges = []
    for step in range(num_merges):
        if not pair_freq:
            break
        best = max(pair_freq, key=lambda p: (pair_freq[p], p))
        if pair_freq[best] < min_freq:
            break
        merges.append(best)
        a, b = best
        merged = a + b
        for i in list(pair_where[best]):
            remove_word_pairs(i)
            sym = words[i]
            out, k = [], 0
            while k < len(sym):
                if k < len(sym) - 1 and sym[k] == a and sym[k + 1] == b:
                    out.append(merged)
                    k += 2
                else:
                    out.append(sym[k])
                    k += 1
            words[i] = out
            add_word_pairs(i)
    return merges


# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

class MorphemeBPE:
    def __init__(self, merges):
        # rank = merge priority (lower = applied first)
        self.ranks = {(a, b): i for i, (a, b) in enumerate(merges)}

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]])

    def encode(self, morphemes):
        """Merge a word's morpheme list into morpheme-BPE tokens."""
        sym = list(morphemes)
        while len(sym) > 1:
            best_rank, best_idx = None, None
            for j in range(len(sym) - 1):
                r = self.ranks.get((sym[j], sym[j + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_idx = r, j
            if best_idx is None:
                break
            sym = sym[:best_idx] + [sym[best_idx] + sym[best_idx + 1]] + sym[best_idx + 2:]
        return sym


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="learn morpheme merges")
    tr.add_argument("input", help="morpheme-segmented corpus (segment_morphemes.py)")
    tr.add_argument("--merges", type=int, default=8000)
    tr.add_argument("--min-freq", type=int, default=2)
    tr.add_argument("-o", "--output", required=True)
    args = ap.parse_args(argv[1:])

    if args.cmd == "train":
        t0 = time.time()
        wf = read_word_freqs(args.input)
        n_base = len({m for w in wf for m in w})
        print(f"{sum(wf.values()):,} word tokens, {len(wf):,} unique words, "
              f"{n_base:,} base morphemes", file=sys.stderr)
        merges = learn_bpe(wf, args.merges, min_freq=args.min_freq)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(
            {"morph_sep": MORPH_SEP, "merges": [list(m) for m in merges]},
            ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"learned {len(merges):,} merges (vocab ≈ {n_base + len(merges):,}) "
              f"in {time.time()-t0:.0f}s -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
