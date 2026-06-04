"""
clean_corpus.py — filter and deduplicate a noisy Turkish web corpus.

The raw export is web-scraped and mixed: English text, phone numbers,
addresses, boilerplate. This streams it line by line and keeps lines that
look like real Turkish prose, using a fast, dependency-free heuristic (no
language-ID model — that would be far too slow at hundreds of millions of
lines). It is intentionally conservative: better to drop a borderline line
than to pollute the tokenizer vocabulary with English/boilerplate.

A line is KEPT when all of these hold:
  - enough words and characters (not a fragment),
  - mostly letters (not digit/punctuation soup: phone numbers, tables),
  - shows a Turkish signal (Turkish-specific letters, or Turkish function
    words), and
  - is not English-dominant.

Optional exact deduplication (post-filter) removes repeated lines.

Output (default data/corpus.clean.txt) is gitignored. Stats and a
drop-reason breakdown go to stderr.

Usage:
    python clean_corpus.py ts_corpus_ver_2-export.txt -o data/corpus.clean.txt
    python clean_corpus.py raw.txt --no-dedup --sample-report 5
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path

import _tok  # noqa: F401
from tr_phonology import tr_lower

# Turkish-specific non-ASCII letters. ASCII I/i are EXCLUDED: they are
# ambiguous across languages, and tr_lower would turn English "I" into
# Turkish "ı", faking a Turkish signal (e.g. "Instructors"). Note that
# İ (U+0130, capital I-with-dot) IS included — it is unambiguously Turkish.
# The set covers both cases of each Turkish-specific vowel/consonant.
_TR_CHARS = set("çğıöşüâîûÇĞİÖŞÜÂÎÛ")
# Single tokens only — these are matched against whitespace-split words.
_TR_STOP = {
    "ve", "bir", "bu", "da", "de", "için", "ile", "ki", "mi", "mı", "mu", "mü",
    "çok", "daha", "en", "ne", "o", "gibi", "kadar", "sonra", "ama", "ya",
    "her", "ben", "sen", "biz", "siz", "onlar", "var", "yok", "olarak",
    "olan", "ise", "değil", "şu", "veya", "ancak", "yani",
}
# A couple of obvious Turkish words that are also short; plus English markers.
_EN_STOP = {
    "the", "and", "of", "to", "in", "is", "are", "you", "for", "on", "with",
    "as", "this", "that", "it", "be", "at", "by", "from", "or", "an", "we",
    "your", "have", "was", "will", "can", "all", "has", "our",
}

MIN_WORDS = 4
MIN_CHARS = 24
MIN_LETTER_RATIO = 0.55
MAX_DIGIT_RATIO = 0.20


def classify(line):
    """Return None to keep, or a short reason string to drop."""
    s = line.strip()
    if len(s) < MIN_CHARS:
        return "short"
    words = s.split()
    if len(words) < MIN_WORDS:
        return "few_words"
    letters = sum(c.isalpha() for c in s)
    if letters < MIN_LETTER_RATIO * len(s):
        return "low_letter_ratio"
    digits = sum(c.isdigit() for c in s)
    if digits > MAX_DIGIT_RATIO * len(s):
        return "digit_heavy"
    lw = [tr_lower(w) for w in words]  # lowercase the already-split tokens
    tr_hits = sum(1 for w in lw if w in _TR_STOP)
    en_hits = sum(1 for w in lw if w in _EN_STOP)
    tr_chars = sum(1 for c in s if c in _TR_CHARS)  # raw text, not lowered
    if tr_chars == 0 and tr_hits == 0:
        return "no_turkish_signal"
    if en_hits >= 2 and en_hits > tr_hits:
        return "english_dominant"
    return None


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="data/corpus.clean.txt")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip exact line dedup (saves memory)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only read the first N input lines (for quick tests)")
    ap.add_argument("--sample-report", type=int, default=0,
                    help="print this many kept + dropped example lines")
    args = ap.parse_args(argv[1:])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_kept = 0
    drops = {}
    seen = set() if not args.no_dedup else None
    n_dup = 0
    kept_samples, dropped_samples = [], []
    t0 = time.time()

    with open(args.input, encoding="utf-8", errors="replace") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            if args.limit and n_in > args.limit:
                n_in -= 1
                break
            reason = classify(line)
            if reason:
                drops[reason] = drops.get(reason, 0) + 1
                if len(dropped_samples) < args.sample_report:
                    dropped_samples.append((reason, line.strip()[:120]))
                continue
            s = " ".join(line.split())  # normalize whitespace
            if seen is not None:
                # Use a 128-bit blake2b digest (32 hex chars) instead of
                # Python's hash() to avoid birthday-paradox collisions on
                # 40M+ lines (hash() has a 64-bit space and is non-uniform).
                h = hashlib.blake2b(s.encode(), digest_size=16).digest()
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
            fout.write(s + "\n")
            n_kept += 1
            if len(kept_samples) < args.sample_report:
                kept_samples.append(s[:120])
            if n_in % 2_000_000 == 0:  # every 2M input lines (not kept lines)
                print(f"  ...{n_in:,} read, {n_kept:,} kept "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr)

    dt = time.time() - t0
    print(f"\n[clean_corpus] read {n_in:,} lines in {dt:.0f}s "
          f"({n_in/dt:.0f} lines/s)", file=sys.stderr)
    print(f"  kept   {n_kept:,} ({100*n_kept/max(n_in,1):.1f}%)", file=sys.stderr)
    if seen is not None:
        print(f"  dup    {n_dup:,} dropped as duplicates", file=sys.stderr)
    print(f"  dropped by reason:", file=sys.stderr)
    for r, c in sorted(drops.items(), key=lambda kv: -kv[1]):
        print(f"    {r:20s} {c:,}", file=sys.stderr)
    print(f"  -> {out_path}", file=sys.stderr)
    if args.sample_report:
        print("\n  KEPT samples:", file=sys.stderr)
        for s in kept_samples:
            print(f"    + {s}", file=sys.stderr)
        print("  DROPPED samples:", file=sys.stderr)
        for r, s in dropped_samples:
            print(f"    - [{r}] {s}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
