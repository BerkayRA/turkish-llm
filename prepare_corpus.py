"""
prepare_corpus.py — build training corpora for the Turkish tokenizer.

Produces two parallel line-aligned corpora from raw Turkish text:
  - <out>/corpus.raw.txt    normalized surface text (attached clitics split,
                            one sentence per line). This is what a normal
                            subword tokenizer trains on.
  - <out>/corpus.morph.txt  morpheme-segmented text: each morpheme is a
                            whitespace token. Training a subword model on
                            THIS biases its merges to respect morpheme
                            boundaries (our "morpheme-aware" experiment).

Normalization (clitic splitting, surface cleanup) is done by the bundled
turkish-tokenizer (vendor/ submodule).

For the smoke test the input is the UD_Turkish-IMST treebank text (small).
For a real run, point --text at a large Turkish corpus (one doc/line).

Usage:
    python prepare_corpus.py --conllu-dir <UD dir> --out data
    python prepare_corpus.py --text big_corpus.txt --out data
"""

import argparse
import sys
from pathlib import Path

import _tok  # noqa: F401  (puts the tokenizer submodule on sys.path)
from tr_api import Tokenizer, TokenizerConfig
import tr_normalize as N

REPO = Path(__file__).resolve().parent


def ud_sentences(conllu_dir):
    marker = "# text = "
    for f in sorted(Path(conllu_dir).glob("*.conllu")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(marker):
                    yield line[len(marker):].strip()


def text_lines(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--conllu-dir",
                     help="UD treebank dir; uses '# text =' sentences")
    src.add_argument("--text", help="plain text corpus, one line per doc")
    ap.add_argument("--out", default="data", help="output directory")
    args = ap.parse_args(argv[1:])

    if not args.conllu_dir and not args.text:
        # Default smoke source: the UD treebank shipped with the tokenizer
        # repo (present in a local checkout; absent in a fresh submodule).
        for cand in (_tok.TOKENIZER_DIR / "UD_Turkish-IMST",
                     REPO.parent / "turkish-tokenizer" / "UD_Turkish-IMST"):
            if cand.exists():
                args.conllu_dir = str(cand)
                break
        if not args.conllu_dir:
            sys.exit("No corpus given and no UD treebank found; pass --text or --conllu-dir")

    sentences = (ud_sentences(args.conllu_dir) if args.conllu_dir
                 else text_lines(args.text))
    print(f"Source: {args.conllu_dir or args.text}", file=sys.stderr)

    tok = Tokenizer(TokenizerConfig(
        suggest_on_oov=False, include_alternatives=False))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "corpus.raw.txt"
    morph_path = out / "corpus.morph.txt"

    n = 0
    with open(raw_path, "w", encoding="utf-8") as raw_f, \
         open(morph_path, "w", encoding="utf-8") as morph_f:
        for s in sentences:
            if not s:
                continue
            result = tok.tokenize_text(
                s, suggest=False, tail_repair=False, alternatives=False)
            raw_f.write(N.render_surface(result, fold=False) + "\n")
            # morpheme tokens separated by spaces (each morpheme a token)
            morph_f.write(N.render_morphemes(result, sep=" ") + "\n")
            n += 1

    print(f"Wrote {n} lines -> {raw_path}, {morph_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
