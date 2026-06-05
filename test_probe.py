"""
test_probe.py — probing harness + benchmark sanity.

Run with:    python -m unittest test_probe -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import probe_eval as PE

BENCH = Path(__file__).resolve().parent / "benchmarks" / "probe.jsonl"


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.pairs = [
            {"category": "a", "good": "G1", "bad": "B1"},
            {"category": "a", "good": "G2", "bad": "B2"},
            {"category": "b", "good": "G3", "bad": "B3"},
        ]

    def test_oracle_scores_100(self):
        good = {p["good"] for p in self.pairs}
        cats = PE.evaluate(self.pairs, lambda t: 1.0 if t in good else 0.0)
        self.assertEqual(cats["a"], [2, 2])
        self.assertEqual(cats["b"], [1, 1])

    def test_inverted_scores_0(self):
        good = {p["good"] for p in self.pairs}
        cats = PE.evaluate(self.pairs, lambda t: 0.0 if t in good else 1.0)
        self.assertEqual(cats["a"], [2, 0])
        self.assertEqual(cats["b"], [1, 0])


class TestCharUnigram(unittest.TestCase):

    def test_scorer_prefers_frequent_chars(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                         encoding="utf-8") as f:
            f.write("aaaa aaaa aaaa\n" * 100)
            path = f.name
        score, name = PE.char_unigram_scorer(path)
        self.assertIn("char_unigram", name)
        # 'a' is far more frequent than 'z' -> higher per-char log-prob
        self.assertGreater(score("aaaa"), score("zzzz"))


class TestBenchmark(unittest.TestCase):
    """The committed probing benchmark is well-formed."""

    @classmethod
    def setUpClass(cls):
        if not BENCH.exists():
            raise unittest.SkipTest("benchmarks/probe.jsonl not present")
        cls.pairs = PE.load_pairs(str(BENCH))

    def test_has_three_categories(self):
        cats = {p["category"] for p in self.pairs}
        self.assertEqual(cats, {"agreement", "possessive", "harmony"})

    def test_pairs_are_minimal_and_distinct(self):
        self.assertGreater(len(self.pairs), 100)
        for p in self.pairs:
            self.assertNotEqual(p["good"], p["bad"])
            self.assertTrue(p["good"] and p["bad"])


if __name__ == "__main__":
    unittest.main()
