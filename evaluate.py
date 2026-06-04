"""
evaluate.py — compare trained tokenizers on Turkish morphological fitness.

Discovers the tokenizers trained by train_tokenizer.py (models/) and scores
each with the bundled tr_fertility metric (reused via score_corpus):

  fertility            subword tokens / word   (lower = better for Turkish)
  tok/morph            subword tokens / morpheme
  single%              words emitted as one token
  boundary P/R/F1      do subword cuts land on true morpheme boundaries?

It also evaluates two baselines (whitespace, char) and a MORPHEME-AWARE
tokenizer: segment each word into morphemes with the analyzer, then subword-
encode each morpheme separately and concatenate. Because pieces never cross
a morpheme boundary, boundary recall is 1.0 by construction — the clearest
way to "leverage" the morphology. The comparison shows what that buys (and
costs) in fertility versus a vanilla subword model.

Usage:
    python evaluate.py --corpus data/corpus.raw.txt --limit 8000
"""

import argparse
import sys
from pathlib import Path

import _tok  # noqa: F401
from tr_api import Tokenizer, TokenizerConfig
import tr_fertility as F

REPO = Path(__file__).resolve().parent


class MorphemeAwareAdapter:
    """Segment a word into morphemes (via the analyzer), subword-encode each
    morpheme with a base SentencePiece model, and concatenate. Guarantees
    every morpheme boundary is a token boundary."""

    def __init__(self, base_spm_path, tok, label=None):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor(model_file=str(base_spm_path))
        self._tok = tok
        self.name = label or f"morph+{Path(base_spm_path).stem}"

    def encode(self, word):
        a = self._tok.tokenize(word, suggest=False, tail_repair=False,
                               alternatives=False, split_clitics=False)
        morphs = ([m["chunk"] for m in a.get("morphemes", [])]
                  if a.get("parsed") else [])
        if not morphs:
            morphs = [word]
        pieces = []
        for m in morphs:
            pieces.extend(self._sp.encode(m, out_type=str))
        return pieces


def discover(models_dir):
    """Yield (label, adapter_factory) for every trained model on disk."""
    md = Path(models_dir)
    for p in sorted(md.glob("sp_*.model")):
        yield p.stem, (lambda p=p: F.SpmAdapter(str(p)))
    for p in sorted(md.glob("byte_bpe_*.json")):
        yield p.stem, (lambda p=p: F.TokenizersAdapter(str(p)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.raw.txt")
    ap.add_argument("--models", default="models")
    ap.add_argument("--limit", type=int, default=8000,
                    help="cap word tokens scored (the analyzer is the cost)")
    args = ap.parse_args(argv[1:])

    tok = Tokenizer(TokenizerConfig(
        suggest_on_oov=False, include_alternatives=False))

    def lines():
        with open(args.corpus, encoding="utf-8") as f:
            for ln in f:
                yield ln

    rows = []

    def run(label, adapter):
        m = F.score_corpus(adapter, lines(), tok, limit=args.limit)
        m["label"] = label
        rows.append(m)
        print(f"  scored {label:22s} fertility={m['fertility']:.3f}", file=sys.stderr)

    # Baselines
    run("whitespace", F.WhitespaceAdapter())
    run("char", F.CharAdapter())
    # Trained models
    largest_unigram = None
    for label, factory in discover(args.models):
        run(label, factory())
        if label.startswith("sp_unigram_"):
            largest_unigram = Path(args.models) / f"{label}.model"
    # Morpheme-aware (segment-then-subword over the largest plain unigram)
    if largest_unigram and largest_unigram.exists():
        run("morpheme-aware", MorphemeAwareAdapter(largest_unigram, tok,
                                                   label="morpheme-aware"))

    # Table
    rows.sort(key=lambda r: r["fertility"])
    hdr = f"{'tokenizer':24s}{'fertility':>10s}{'tok/morph':>11s}{'single%':>9s}{'bound-F1':>10s}{'recall':>8s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:24s}{r['fertility']:>10.3f}{r['tokens_per_morpheme']:>11.3f}"
              f"{r['single_token_pct']:>8.1f}%{r['boundary_f1']:>10.3f}{r['boundary_recall']:>8.3f}")
    print(f"\n(corpus: {args.corpus}, up to {args.limit} words; "
          f"lower fertility better, higher boundary-F1/recall = more "
          f"morpheme-aligned)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
