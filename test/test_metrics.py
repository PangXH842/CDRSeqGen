import unittest

from cdrseqgen.eval.metrics import (
    LENGTH_BUCKETS,
    amino_acid_recovery,
    calculate,
    edit_similarity,
    hamming_fraction,
    mean_pairwise_diversity,
    per_position_entropy,
    unique_fraction,
)


class SharedMetricPrimitiveTests(unittest.TestCase):
    """These primitives are shared by the native evaluators and benchmark/scorer.py, so a defect
    here changes both sides of a matched comparison identically — and therefore invisibly.

    Unequal-length inputs get their own cases on purpose: the length-free track is the only place
    they arise, and every aggregate-level fixture in the suite uses equal-length candidates, where
    several plausible wrong formulas coincide with the right one.
    """

    def test_hamming_normalises_by_the_longer_sequence(self):
        self.assertEqual(hamming_fraction("AAAA", "AAAA"), 0.0)
        self.assertEqual(hamming_fraction("AAAA", "CCCC"), 1.0)
        self.assertEqual(hamming_fraction("AACC", "AAAA"), 0.5)

    def test_hamming_counts_a_length_difference_as_mismatches(self):
        # 4 shared positions all match; the 4 extra residues are 4 differences out of 8.
        self.assertEqual(hamming_fraction("AAAA", "AAAACCCC"), 0.5)
        # Dividing by the overlap instead of the longer length would give 1.0 here, not 0.5.
        self.assertEqual(hamming_fraction("AAAA", "AAAAAAAA"), 0.5)
        self.assertEqual(hamming_fraction("", "AAAA"), 1.0)
        self.assertEqual(hamming_fraction("", ""), 0.0)

    def test_diversity_is_the_mean_over_unordered_pairs(self):
        self.assertEqual(mean_pairwise_diversity(["AAAA"]), 0.0)
        self.assertEqual(mean_pairwise_diversity(["AAAA", "AAAA"]), 0.0)
        self.assertEqual(mean_pairwise_diversity(["AAAA", "CCCC"]), 1.0)
        # unequal lengths must not be silently skipped or over-weighted
        self.assertEqual(mean_pairwise_diversity(["AAAA", "AAAAAAAA"]), 0.5)

    def test_edit_similarity_is_defined_across_lengths(self):
        self.assertEqual(edit_similarity("ACDEF", "ACDEF"), 1.0)
        self.assertEqual(edit_similarity("", ""), 1.0)
        self.assertEqual(edit_similarity("ACDEF", "ACDE"), 0.8)      # one deletion of five
        self.assertEqual(edit_similarity("ACDEF", "ACDEFG"), 5 / 6)  # one insertion of six
        self.assertAlmostEqual(edit_similarity("ACDEF", "AXDEF"), 0.8)

    def test_entropy_is_none_unless_every_candidate_shares_a_length(self):
        self.assertIsNone(per_position_entropy([]))
        self.assertIsNone(per_position_entropy(["AC", "ACD"]))
        self.assertIsNone(per_position_entropy([""]))
        self.assertEqual(per_position_entropy(["AC", "AC"]), 0.0)
        self.assertEqual(per_position_entropy(["AA", "CC"]), 1.0)  # 1 bit at each position

    def test_recovery_is_normalised_by_the_native_length(self):
        self.assertEqual(amino_acid_recovery("ACDE", "ACDE"), 1.0)
        self.assertEqual(amino_acid_recovery("ACDE", "ACDX"), 0.75)
        # a short prediction is penalised for what it failed to produce
        self.assertEqual(amino_acid_recovery("AC", "ACDE"), 0.5)
        self.assertIsNone(amino_acid_recovery("ACDE", ""))

    def test_unique_fraction_counts_distinct_draws(self):
        self.assertEqual(unique_fraction([]), 0.0)
        self.assertEqual(unique_fraction(["A", "A", "A"]), 1 / 3)
        self.assertEqual(unique_fraction(["A", "C"]), 1.0)

    def test_length_buckets_partition_every_loop_length(self):
        for length in range(0, 60):
            hits = [name for name, lo, hi in LENGTH_BUCKETS if lo <= length <= hi]
            self.assertEqual(len(hits), 1, f"length {length} matched {hits}")


class CalculateBlockTests(unittest.TestCase):
    def test_positional_recovery_is_conditioned_on_a_length_match(self):
        """The length-free contract: AAR is defined only for candidates of the native length,
        while length error and edit similarity cover every candidate."""
        out = calculate(["ACDE", "ACD", "ACDEF"], native="ACDE")

        self.assertEqual(out["length_match_rate"], 1 / 3)
        self.assertAlmostEqual(out["length_absolute_error_mean"], 2 / 3)
        self.assertEqual(out["aar_mean"], 1.0)            # only the length-matched draw scores
        self.assertEqual(out["aar_length_matched_mean"], 1.0)
        self.assertEqual(out["novelty_mean"], 0.0)
        self.assertIn("edit_similarity_mean", out)         # all three contribute here
        self.assertIsNone(out["per_position_entropy_bits"])  # lengths differ

    def test_no_native_reports_design_metrics_only(self):
        out = calculate(["ACDE", "ACDF"])
        self.assertNotIn("aar_mean", out)
        self.assertNotIn("length_match_rate", out)
        self.assertEqual(out["n_samples"], 2)
        self.assertEqual(out["unique_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
