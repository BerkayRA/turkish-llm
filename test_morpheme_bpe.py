"""
test_morpheme_bpe.py — from-scratch morpheme-BPE learn + encode.

Run with:    python -m unittest test_morpheme_bpe -v
"""

import unittest

import morpheme_bpe as M


class TestLearnBPE(unittest.TestCase):

    def setUp(self):
        # gel+iyor is the most frequent adjacent morpheme pair (18).
        self.wf = {
            ("gel", "iyor", "um"): 10,
            ("gel", "iyor", "sun"): 8,
            ("kitab", "lar", "ı"): 4,
            ("kitab", "ı"): 5,
            ("ev",): 7,
        }

    def test_most_frequent_pair_merged_first(self):
        merges = M.learn_bpe(self.wf, num_merges=10, min_freq=2)
        self.assertEqual(merges[0], ("gel", "iyor"))

    def test_min_freq_stops_merging(self):
        # Nothing occurs >= 100 times, so no merges are learned.
        self.assertEqual(M.learn_bpe(self.wf, num_merges=10, min_freq=100), [])

    def test_merge_count_capped(self):
        self.assertLessEqual(len(M.learn_bpe(self.wf, num_merges=3)), 3)


class TestEncode(unittest.TestCase):

    def setUp(self):
        wf = {("gel", "iyor", "um"): 10, ("gel", "iyor", "sun"): 8,
              ("kitab", "lar", "ı"): 4, ("kitab", "ı"): 5}
        self.bpe = M.MorphemeBPE(M.learn_bpe(wf, num_merges=10, min_freq=2))

    def test_frequent_word_collapses(self):
        self.assertEqual(self.bpe.encode(["gel", "iyor", "um"]), ["geliyorum"])

    def test_single_morpheme_unchanged(self):
        self.assertEqual(self.bpe.encode(["ev"]), ["ev"])

    def test_tokens_reconstruct_the_word(self):
        # Every token is a sequence of whole morphemes -> concatenation is
        # the surface, and token boundaries are a subset of morpheme ones.
        for morphs in (["gel", "iyor", "um"], ["kitab", "lar", "ı"], ["a", "b", "c"]):
            toks = self.bpe.encode(morphs)
            self.assertEqual("".join(toks), "".join(morphs))

    def test_unknown_morphemes_pass_through(self):
        # No applicable merge -> returned as-is.
        self.assertEqual(self.bpe.encode(["xyz", "qrs"]), ["xyz", "qrs"])


if __name__ == "__main__":
    unittest.main()
