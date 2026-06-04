"""
train_tokenizer.py — train candidate subword tokenizers for a Turkish LLM.

Trains, from the prepared corpora (see prepare_corpus.py), three families so
they can be compared with evaluate.py / tr_fertility:

  sp_unigram_<V>   SentencePiece Unigram on raw text   (the standard baseline)
  byte_bpe_<V>     byte-level BPE on raw text           (GPT/Llama-style)
  sp_morph_<V>     SentencePiece Unigram on MORPHEME-segmented text
                   — the morpheme-aware experiment: training on text whose
                   morpheme boundaries are word boundaries biases the learned
                   pieces to align with morphology.

Tokenizer training is CPU-only and fast; GPUs are not used here. Vocab sizes
are a free parameter — pass real sizes (e.g. 16000 32000 48000) on a real
corpus. The small defaults only suit the tiny UD smoke corpus.

Usage:
    python train_tokenizer.py --vocab-sizes 16000 32000 --byte-bpe 32000
"""

import argparse
import sys
from pathlib import Path

import sentencepiece as spm
from tokenizers import ByteLevelBPETokenizer

REPO = Path(__file__).resolve().parent
SPECIALS = ["<pad>", "<unk>", "<s>", "</s>"]


def train_sentencepiece(corpus, model_prefix, vocab_size, model_type="unigram",
                        input_sentence_size=0):
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=0.9995,      # cover (nearly) all Turkish characters
        byte_fallback=True,             # never emit <unk>
        hard_vocab_limit=False,         # tiny corpora may not reach vocab_size
        # On a large corpus, train from a shuffled sample of this many
        # sentences (0 = use all). A few million is plenty for a 32k+ vocab.
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        normalization_rule_name="nfkc",
    )


def train_byte_bpe(corpus, out_path, vocab_size):
    tk = ByteLevelBPETokenizer()
    tk.train(files=[str(corpus)], vocab_size=vocab_size,
             min_frequency=2, special_tokens=SPECIALS)
    tk.save(str(out_path))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/corpus.raw.txt")
    ap.add_argument("--morph", default="data/corpus.morph.txt")
    ap.add_argument("--out", default="models")
    ap.add_argument("--vocab-sizes", type=int, nargs="+", default=[2000, 4000, 8000])
    ap.add_argument("--byte-bpe", type=int, default=4000,
                    help="byte-level BPE vocab size (0 to skip)")
    ap.add_argument("--no-morph", action="store_true",
                    help="skip the morpheme-aware variant")
    ap.add_argument("--input-sentence-size", type=int, default=0,
                    help="SentencePiece: train from a shuffled sample of this "
                         "many sentences (0 = all; use a few million on a "
                         "large corpus)")
    args = ap.parse_args(argv[1:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw, morph = Path(args.raw), Path(args.morph)
    if not raw.exists():
        sys.exit(f"raw corpus not found: {raw} — run clean_corpus.py / "
                 f"prepare_corpus.py first")
    trained = []

    for v in args.vocab_sizes:
        try:
            train_sentencepiece(raw, out / f"sp_unigram_{v}", v,
                                input_sentence_size=args.input_sentence_size)
            trained.append(f"sp_unigram_{v}.model")
            print(f"[ok] sp_unigram_{v}", file=sys.stderr)
        except RuntimeError as e:
            # SentencePiece raises RuntimeError when the corpus is too small
            # to fill the requested vocab or the sentence limit is too tight.
            # Any other exception (OOM, disk full, etc.) propagates upward.
            print(f"[skip] sp_unigram_{v}: {e}", file=sys.stderr)
        if not args.no_morph and morph.exists():
            try:
                train_sentencepiece(morph, out / f"sp_morph_{v}", v,
                                    input_sentence_size=args.input_sentence_size)
                trained.append(f"sp_morph_{v}.model")
                print(f"[ok] sp_morph_{v}", file=sys.stderr)
            except RuntimeError as e:
                # Same rationale as sp_unigram above.
                print(f"[skip] sp_morph_{v}: {e}", file=sys.stderr)

    if args.byte_bpe:
        v = args.byte_bpe
        try:
            train_byte_bpe(raw, out / f"byte_bpe_{v}.json", v)
            trained.append(f"byte_bpe_{v}.json")
            print(f"[ok] byte_bpe_{v}", file=sys.stderr)
        except (RuntimeError, ValueError) as e:
            # ByteLevelBPETokenizer raises ValueError when vocab_size
            # exceeds unique byte sequences in a tiny corpus.
            print(f"[skip] byte_bpe_{v}: {e}", file=sys.stderr)

    print(f"\nTrained {len(trained)} tokenizers in {out}/:", file=sys.stderr)
    for t in trained:
        print(f"  {t}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
