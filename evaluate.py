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
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import _tok  # noqa: F401
from tr_api import Tokenizer, TokenizerConfig
import tr_fertility as F

REPO = Path(__file__).resolve().parent


def describe(label):
    """(algorithm, vocab_size) for a tokenizer label. vocab_size is None for
    baselines / the inference-time morpheme-aware tokenizer."""
    if label in ("whitespace", "char"):
        return (f"{label} (baseline)", None)
    if label == "morpheme-aware":
        return ("morpheme-aware (segment-then-subword)", None)
    if label.startswith("sp_unigram_"):
        return ("SentencePiece Unigram", int(label.rsplit("_", 1)[1]))
    if label.startswith("sp_morph_"):
        return ("SentencePiece Unigram, morpheme-trained", int(label.rsplit("_", 1)[1]))
    if label.startswith("byte_bpe_"):
        return ("byte-level BPE", int(label.rsplit("_", 1)[1]))
    return (label, None)


def corpus_word_count(path):
    """Fast whitespace word count over the whole corpus (for the report)."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n += len(line.split())
    return n


def commentary(rows):
    """Auto-generated commentary on relative performance. `rows` are metric
    dicts (with 'label'); order doesn't matter here."""
    out = []
    real = [r for r in rows if r["label"] not in ("whitespace", "char")]
    if not real:
        return "No trained tokenizers to compare."
    best_fert = min(real, key=lambda r: r["fertility"])
    best_f1 = max(rows, key=lambda r: r["boundary_f1"])
    best_rec = max(rows, key=lambda r: r["boundary_recall"])
    out.append(f"- **Lowest fertility (best compression):** `{best_fert['label']}` "
               f"at {best_fert['fertility']:.3f} tokens/word.")
    out.append(f"- **Best morpheme alignment (boundary F1):** `{best_f1['label']}` "
               f"at F1 {best_f1['boundary_f1']:.3f} (recall {best_rec['boundary_recall']:.3f} "
               f"for `{best_rec['label']}`).")
    # Vocab-size effect within the plain SentencePiece Unigram family.
    uni = sorted((r for r in real if r["label"].startswith("sp_unigram_")),
                 key=lambda r: describe(r["label"])[1] or 0)
    if len(uni) >= 2:
        lo, hi = uni[0], uni[-1]
        out.append(f"- **Vocab-size effect:** raising the SentencePiece Unigram vocab "
                   f"{describe(lo['label'])[1]}→{describe(hi['label'])[1]} moved fertility "
                   f"{lo['fertility']:.3f}→{hi['fertility']:.3f} "
                   f"({'down' if hi['fertility'] < lo['fertility'] else 'up'}).")
    # Morpheme-aware vs the plain baseline it's built on.
    ma = next((r for r in rows if r["label"] == "morpheme-aware"), None)
    if ma and uni:
        base = uni[-1]
        out.append(f"- **Morpheme-awareness tradeoff:** the segment-then-subword tokenizer "
                   f"reaches boundary recall {ma['boundary_recall']:.3f} "
                   f"(vs {base['boundary_recall']:.3f} for `{base['label']}`) "
                   f"at fertility {ma['fertility']:.3f} (vs {base['fertility']:.3f}) — "
                   f"more tokens, but morpheme-aligned by construction.")
    morph = sorted((r for r in real if r["label"].startswith("sp_morph_")),
                   key=lambda r: describe(r["label"])[1] or 0)
    if morph and uni:
        m, u = morph[-1], uni[-1]
        out.append(f"- **Training on morpheme-segmented text** (`{m['label']}`) lifts boundary "
                   f"recall to {m['boundary_recall']:.3f} (vs {u['boundary_recall']:.3f} for the "
                   f"raw-trained `{u['label']}`) at fertility {m['fertility']:.3f} vs {u['fertility']:.3f}.")
    return "\n".join(out)


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


def build_report(rows, meta):
    """Return (markdown, json_obj) for a detailed evaluation report."""
    ranked = sorted(rows, key=lambda r: r["fertility"])
    enriched = []
    for r in ranked:
        algo, vocab = describe(r["label"])
        enriched.append({**r, "algorithm": algo, "vocab_size": vocab})

    md = []
    md.append("# Tokenizer Evaluation Report\n")
    md.append(f"- **Generated:** {meta['timestamp']}")
    md.append(f"- **Corpus:** `{meta['corpus']}` — {meta['corpus_words']:,} words total, "
              f"{meta['scored_words']:,} scored (limit {meta['limit']})")
    md.append(f"- **Morphological reference:** turkish-tokenizer "
              f"({meta['analyzer_morphemes_per_word']:.3f} morphemes/word on the scored slice)")
    md.append(f"- **Tokenizers compared:** {len(rows)}\n")
    md.append("## Results (sorted by fertility, low → high)\n")
    md.append("| tokenizer | algorithm | vocab | fertility | tok/morph | single% | "
              "bound-P | bound-R | bound-F1 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in enriched:
        md.append(f"| `{r['label']}` | {r['algorithm']} | "
                  f"{r['vocab_size'] if r['vocab_size'] else '—'} | "
                  f"{r['fertility']:.3f} | {r['tokens_per_morpheme']:.3f} | "
                  f"{r['single_token_pct']:.1f}% | {r['boundary_precision']:.3f} | "
                  f"{r['boundary_recall']:.3f} | {r['boundary_f1']:.3f} |")
    md.append("\n## Commentary\n")
    md.append(commentary(rows))
    md.append("\n## Metric definitions\n")
    md.append("- **fertility** — subword tokens per word; lower is better (less "
              "compute, longer effective context).")
    md.append("- **tok/morph** — subword tokens per morpheme (analyzer reference).")
    md.append("- **single%** — words emitted as exactly one token.")
    md.append("- **boundary P/R/F1** — do subword cut points fall on true morpheme "
              "boundaries? Recall = fraction of morpheme boundaries captured; "
              "precision = fraction of subword cuts that are morpheme boundaries.")
    md.append("\nSee `docs/METRICS.md` for full definitions and interpretation.\n")

    obj = {
        "meta": meta,
        "results": [{
            "label": r["label"], "algorithm": r["algorithm"],
            "vocab_size": r["vocab_size"],
            "fertility": r["fertility"],
            "tokens_per_morpheme": r["tokens_per_morpheme"],
            "morphemes_per_word": r["morphemes_per_word"],
            "single_token_pct": r["single_token_pct"],
            "boundary_precision": r["boundary_precision"],
            "boundary_recall": r["boundary_recall"],
            "boundary_f1": r["boundary_f1"],
            "aligned_pct": r["aligned_pct"],
            "words": r["words"], "subword_tokens": r["subword_tokens"],
        } for r in enriched],
    }
    return "\n".join(md), obj


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="data/corpus.raw.txt")
    ap.add_argument("--models", default="models")
    ap.add_argument("--limit", type=int, default=8000,
                    help="cap word tokens scored (the analyzer is the cost)")
    ap.add_argument("--report-dir", default="reports",
                    help="where detailed reports are written")
    ap.add_argument("--no-report", action="store_true",
                    help="skip writing the markdown+JSON report (on by default)")
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

    # Detailed report after every evaluation (default behaviour).
    if not args.no_report:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        scored = rows[0]["words"] if rows else 0
        mpw = (sum(r["morphemes_per_word"] for r in rows) / len(rows)) if rows else 0.0
        meta = {
            "timestamp": ts,
            "corpus": args.corpus,
            "corpus_words": corpus_word_count(args.corpus),
            "scored_words": scored,
            "limit": args.limit,
            "analyzer_morphemes_per_word": mpw,
        }
        md, obj = build_report(rows, meta)
        rdir = Path(args.report_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        md_path = rdir / f"eval_{ts}.md"
        json_path = rdir / f"eval_{ts}.json"
        md_path.write_text(md + "\n", encoding="utf-8")
        json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        (rdir / "latest.md").write_text(md + "\n", encoding="utf-8")
        print(f"\nReport written: {md_path}  (+ {json_path.name}, latest.md)",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
