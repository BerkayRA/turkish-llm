"""
probe_dataset.py — generate a Turkish morphological probing set.

Builds minimal pairs (a grammatical sentence vs a minimally-ungrammatical
one) using the morphological generator as gold, BLiMP/CLAMS-style. A model
that has learned Turkish morphology should assign higher probability to the
grammatical member; accuracy is measured by probe_eval.py. Random = 50%.

Categories:
  agreement   subject pronoun vs verb person  (ben geldim / *ben geldin)
  possessive  possessor pronoun vs poss suffix (benim kitabım / *benim kitabın)
  harmony     correct vs vowel-harmony-violating inflection (evde / *evda)

The "good" forms come from tr_generate; the "bad" forms swap in a wrong
agreement/possessive suffix or a harmony-violating vowel, and are verified
against the analyzer so they are genuinely ill-formed (not an accidental
real word). Output: JSONL of {category, good, bad, note}.

Usage:
    python probe_dataset.py -o data/probe.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import _tok  # noqa: F401
from tr_inventory import load_inventory
from tr_morphotactics import load_graph
from tr_lexicon import load_lexicon
from tr_parse import Parser, ParseConfig
import tr_generate as G

# Curated common stems (kept small and high-frequency to avoid OOV noise).
VERBS = ["gel", "git", "gör", "yap", "al", "ver", "bil", "oku", "yaz", "sor",
         "sev", "koş", "dur", "kal", "bul", "bak", "anla", "iste", "söyle", "çalış"]
NOUNS = ["ev", "kitap", "masa", "araba", "okul", "kalem", "yol", "göz", "kapı",
         "çocuk", "ağaç", "kuş", "deniz", "şehir", "renk", "dağ", "yıl", "gün",
         "söz", "el"]

PERSONS = ["1SG", "2SG", "1PL", "2PL"]
SUBJ_PRON = {"1SG": "ben", "2SG": "sen", "1PL": "biz", "2PL": "siz"}
VERB_AGR = {"1SG": "1SG_K", "2SG": "2SG_K", "1PL": "1PL_K", "2PL": "2PL_K"}
POSS_PRON = {"1SG": "benim", "2SG": "senin", "1PL": "bizim", "2PL": "sizin"}
POSS_SUF = {"1SG": "POSS_1SG", "2SG": "POSS_2SG", "1PL": "POSS_1PL", "2PL": "POSS_2PL"}
CASE_SUFFIXES = ["LOC", "DAT", "ABL", "PLUR"]

VOWEL_SWAP = {"a": "e", "e": "a", "ı": "i", "i": "ı",
              "o": "ö", "ö": "o", "u": "ü", "ü": "u"}


def swap_last_vowel(word):
    for i in range(len(word) - 1, -1, -1):
        if word[i] in VOWEL_SWAP:
            return word[:i] + VOWEL_SWAP[word[i]] + word[i + 1:]
    return word


class Builder:
    def __init__(self):
        here = _tok.TOKENIZER_DIR
        self.inv = load_inventory(str(here / "inventory.json"))
        self.graph = load_graph(str(here / "morphotactics.json"))
        self.lex = load_lexicon(str(here / "lexicon_full.json"))
        self.parser = Parser(self.lex, self.inv, self.graph, ParseConfig())

    def gen(self, stem, chain, wc):
        try:
            return G.generate(stem, chain, self.inv, word_class=wc,
                              graph=self.graph, strict=True).surface
        except Exception:
            return None

    def parses_as(self, surface, root, suffix_ids):
        """True if `surface`'s top parse has this root and exactly these
        non-root suffix ids — used to confirm a 'bad' form is NOT actually
        the intended-correct analysis."""
        an = self.parser.parse(surface)
        if not an:
            return False
        top = an[0]
        ids = tuple(m.suffix_id for m in top.morphemes[1:] if m.suffix_id)
        return top.root == root and ids == tuple(suffix_ids)

    def agreement(self):
        items = []
        for v in VERBS:
            for p in PERSONS:
                good_v = self.gen(v, ["PAST", VERB_AGR[p]], "VERB")
                wrong_p = PERSONS[(PERSONS.index(p) + 1) % len(PERSONS)]
                bad_v = self.gen(v, ["PAST", VERB_AGR[wrong_p]], "VERB")
                if not good_v or not bad_v or good_v == bad_v:
                    continue
                items.append({
                    "category": "agreement",
                    "good": f"{SUBJ_PRON[p]} {good_v}",
                    "bad": f"{SUBJ_PRON[p]} {bad_v}",
                    "note": f"{p} subject; verb {p} vs {wrong_p}",
                })
        return items

    def possessive(self):
        items = []
        for n in NOUNS:
            for p in PERSONS:
                good_n = self.gen(n, [POSS_SUF[p]], "NOUN")
                wrong_p = PERSONS[(PERSONS.index(p) + 1) % len(PERSONS)]
                bad_n = self.gen(n, [POSS_SUF[wrong_p]], "NOUN")
                if not good_n or not bad_n or good_n == bad_n:
                    continue
                items.append({
                    "category": "possessive",
                    "good": f"{POSS_PRON[p]} {good_n}",
                    "bad": f"{POSS_PRON[p]} {bad_n}",
                    "note": f"{p} possessor; noun {p} vs {wrong_p}",
                })
        return items

    def harmony(self):
        items = []
        for n in NOUNS:
            for suf in CASE_SUFFIXES:
                good = self.gen(n, [suf], "NOUN")
                if not good:
                    continue
                bad = swap_last_vowel(good)
                if bad == good:
                    continue
                # The bad form must NOT parse as n+suf (i.e. genuinely wrong).
                if self.parses_as(bad, n, [suf]):
                    continue
                items.append({
                    "category": "harmony",
                    "good": good,
                    "bad": bad,
                    "note": f"{n}+{suf}; harmony violation",
                })
        return items


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default="benchmarks/probe.jsonl")
    args = ap.parse_args(argv[1:])

    b = Builder()
    items = b.agreement() + b.possessive() + b.harmony()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    by_cat = {}
    for it in items:
        by_cat[it["category"]] = by_cat.get(it["category"], 0) + 1
    print(f"wrote {len(items)} minimal pairs -> {out}", file=sys.stderr)
    for c, n in sorted(by_cat.items()):
        print(f"  {c:12s} {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
