"""
probe_eval.py — score a language model on the morphological probing set.

Minimal-pair accuracy: for each (good, bad) pair the model is "correct" when
it assigns higher probability to the grammatical member. Per-category and
overall accuracy are reported (random = 50%). A model that has genuinely
learned Turkish morphology scores well above chance; a model with no
morphological knowledge (e.g. the char-unigram baseline) sits near 50%.

This is the harness for the tokenizer A/B: train two small LMs (Unigram-32k
vs morpheme-BPE), wrap each as a scorer, and compare probing accuracy.

Scorers (a scorer maps text -> a log-probability; higher = more likely):
  char_unigram   stdlib baseline char-unigram LM (no morphology -> ~chance)
  hf             a Hugging Face causal LM (needs torch+transformers); the
                 template you'll reuse for our own model

A detailed report (md + json) is written after every run.

Usage:
    python probe_eval.py --scorer char_unigram --corpus data/corpus.train.txt
    python probe_eval.py --scorer hf --hf-model <name-or-dir>
"""

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# -----------------------------------------------------------------------------
# Scorers
# -----------------------------------------------------------------------------

def char_unigram_scorer(corpus_path, max_lines=200000):
    """A character-unigram log-likelihood model (add-1 smoothed). It has no
    notion of morphology, so it is a sanity floor — it should score ~50% on
    the morphological pairs."""
    counts = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            counts.update(line.strip())
    total = sum(counts.values())
    V = len(counts)
    logp = {c: math.log((counts[c] + 1) / (total + V)) for c in counts}
    floor = math.log(1 / (total + V))

    def score(text):
        return sum(logp.get(c, floor) for c in text)
    return score, f"char_unigram({Path(corpus_path).name})"


def hf_scorer(name_or_dir):
    """Causal-LM total log-likelihood via Hugging Face transformers."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("--scorer hf needs `torch` and `transformers`")
    tok = AutoTokenizer.from_pretrained(name_or_dir)
    model = AutoModelForCausalLM.from_pretrained(name_or_dir)
    model.eval()

    def score(text):
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model(**ids, labels=ids["input_ids"])
        n = ids["input_ids"].shape[1]
        return -out.loss.item() * n          # total log-likelihood
    return score, f"hf:{name_or_dir}"


# -----------------------------------------------------------------------------
# Evaluation + report
# -----------------------------------------------------------------------------

def evaluate(pairs, score):
    cats = {}
    for p in pairs:
        c = p["category"]
        good, bad = score(p["good"]), score(p["bad"])
        ok = good > bad
        agg = cats.setdefault(c, [0, 0])      # [n, correct]
        agg[0] += 1
        agg[1] += 1 if ok else 0
    return cats


def build_report(cats, meta):
    total_n = sum(v[0] for v in cats.values())
    total_c = sum(v[1] for v in cats.values())
    md = ["# Morphological Probing Report\n"]
    md.append(f"- **Generated:** {meta['timestamp']}")
    md.append(f"- **Scorer:** {meta['scorer']}")
    md.append(f"- **Dataset:** `{meta['dataset']}` ({total_n} minimal pairs)")
    md.append(f"- **Overall accuracy:** {100*total_c/total_n:.1f}% (random = 50%)\n")
    md.append("## By category\n")
    md.append("| category | pairs | accuracy |")
    md.append("|---|---:|---:|")
    for c in sorted(cats):
        n, ok = cats[c]
        md.append(f"| {c} | {n} | {100*ok/n:.1f}% |")
    md.append("\n## Notes\n")
    md.append("- Accuracy = fraction of pairs where the model assigns higher "
              "probability to the grammatical sentence. 50% = chance.")
    md.append("- A char-unigram baseline near 50% confirms the pairs require "
              "real morphological knowledge, not surface statistics.")
    obj = {
        "meta": meta,
        "overall_accuracy": total_c / total_n if total_n else 0.0,
        "by_category": {c: {"pairs": cats[c][0],
                            "accuracy": cats[c][1] / cats[c][0]} for c in cats},
    }
    return "\n".join(md), obj


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="benchmarks/probe.jsonl")
    ap.add_argument("--scorer", choices=["char_unigram", "hf"], default="char_unigram")
    ap.add_argument("--corpus", default="data/corpus.train.txt",
                    help="corpus for the char_unigram baseline")
    ap.add_argument("--hf-model", help="model name/dir for --scorer hf")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args(argv[1:])

    pairs = load_pairs(args.dataset)
    if args.scorer == "hf":
        if not args.hf_model:
            sys.exit("--scorer hf requires --hf-model")
        score, name = hf_scorer(args.hf_model)
    else:
        score, name = char_unigram_scorer(args.corpus)

    cats = evaluate(pairs, score)
    total_n = sum(v[0] for v in cats.values())
    total_c = sum(v[1] for v in cats.values())
    print(f"scorer: {name}")
    print(f"overall: {100*total_c/total_n:.1f}%  ({total_n} pairs)")
    for c in sorted(cats):
        n, ok = cats[c]
        print(f"  {c:12s} {100*ok/n:5.1f}%  ({n})")

    if not args.no_report:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        meta = {"timestamp": ts, "scorer": name, "dataset": args.dataset}
        md, obj = build_report(cats, meta)
        rdir = Path(args.report_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / f"probe_{ts}.md").write_text(md + "\n", encoding="utf-8")
        (rdir / f"probe_{ts}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (rdir / "probe_latest.md").write_text(md + "\n", encoding="utf-8")
        print(f"\nReport: {rdir}/probe_{ts}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
