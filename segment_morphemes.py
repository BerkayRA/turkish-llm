"""
segment_morphemes.py — morpheme-segment a corpus sample, in parallel.

The morphological analyzer runs at ~115 words/s single-threaded, so
segmenting a useful sample (a few million words) needs multiprocessing.
Each worker builds its own Tokenizer once, then segments lines; output is
morpheme-segmented text (each morpheme a whitespace token), one line in →
one line out (blank / word-less lines dropped).

This produces the training data for the morpheme-BPE tokenizer
(morpheme_bpe.py). A few hundred K–few M words is plenty to learn merges.

Usage:
    python segment_morphemes.py data/corpus.clean.txt -o data/sample.morph.txt \
        --max-lines 150000 --workers 8
"""

import argparse
import itertools
import multiprocessing as mp
import sys
import time
from pathlib import Path

import _tok  # noqa: F401
from tr_api import Tokenizer, TokenizerConfig
import tr_normalize as N

_TOK = None


def _init_worker():
    global _TOK
    _TOK = Tokenizer(TokenizerConfig(
        suggest_on_oov=False, include_alternatives=False))


# Within-word morpheme separator. Word boundaries stay as spaces so the
# morpheme-BPE trainer can merge morphemes WITHIN a word but never across
# word boundaries. e.g. "geliyorum kitabı" -> "gel▁iyor▁um kitab▁ı".
_MORPH_SEP = "▁"


def _segment(line):
    line = line.strip()
    if not line:
        return None
    result = _TOK.tokenize_text(
        line, suggest=False, tail_repair=False, alternatives=False)
    if not any(t["kind"] == "word" for t in result["tokens"]):
        return None
    return N.render_morphemes(result, sep=_MORPH_SEP)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="data/sample.morph.txt")
    ap.add_argument("--max-lines", type=int, default=150000)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = all CPUs")
    args = ap.parse_args(argv[1:])

    workers = args.workers or mp.cpu_count()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def lines():
        with open(args.input, encoding="utf-8") as f:
            for ln in (itertools.islice(f, args.max_lines) if args.max_lines else f):
                yield ln

    n_in = n_out = 0
    t0 = time.time()
    with mp.Pool(workers, initializer=_init_worker) as pool, \
         open(out_path, "w", encoding="utf-8") as fout:
        for seg in pool.imap(_segment, lines(), chunksize=500):
            n_in += 1
            if seg:
                fout.write(seg + "\n")
                n_out += 1
            if n_in % 50000 == 0:
                print(f"  ...{n_in:,} lines ({time.time()-t0:.0f}s)", file=sys.stderr)

    dt = time.time() - t0
    print(f"[segment_morphemes] {n_in:,} lines -> {n_out:,} segmented "
          f"in {dt:.0f}s on {workers} workers -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
