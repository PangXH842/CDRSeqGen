import unittest

from cdrseqgen.eval.metrics import (
    core_anchor_indices,
    recovery_at_indices,
    LENGTH_BUCKETS,
    amino_acid_recovery,
    calculate,
    edit_similarity,
    hamming_fraction,
    mean_pairwise_diversity,
    per_position_entropy,
    position_recovery,
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


class CoreAnchorSplitTests(unittest.TestCase):
    """The one definition of "core" shared by training metrics, generative eval, and the scorer."""

    def test_split_trims_the_requested_ends(self):
        core, anchor = core_anchor_indices(10, 2)
        self.assertEqual(core, [2, 3, 4, 5, 6, 7])
        self.assertEqual(anchor, [0, 1, 8, 9])

    def test_short_loop_is_entirely_anchor(self):
        # Matches recovery._core_anchor_masks: a loop of <= 2*trim residues has no core at all,
        # which is why L2 (mostly 3-mers) reports core recovery over almost no positions.
        for length in (3, 4):
            core, anchor = core_anchor_indices(length, 2)
            self.assertEqual(core, [])
            self.assertEqual(anchor, list(range(length)))

    def test_the_two_halves_partition_the_loop(self):
        for length in range(1, 20):
            core, anchor = core_anchor_indices(length, 2)
            self.assertEqual(sorted(core + anchor), list(range(length)))
            self.assertFalse(set(core) & set(anchor))

    def test_recovery_at_indices_scores_only_the_subset(self):
        native, sample = "ABCDEF", "ABQQEF"
        core, anchor = core_anchor_indices(6, 2)
        self.assertEqual(recovery_at_indices(sample, native, core), 0.0)
        self.assertEqual(recovery_at_indices(sample, native, anchor), 1.0)

    def test_recovery_over_an_empty_subset_is_none_not_zero(self):
        # A loop with no core must report "not measured", never a 0.0 that averages into the mean.
        self.assertIsNone(recovery_at_indices("ABC", "ABC", []))

    def test_position_recovery_scores_each_position_over_length_matched_samples(self):
        curve = position_recovery(["ABQ", "AXQ"], "ABC")
        self.assertEqual([point["recovery"] for point in curve], [1.0, 0.5, 0.0])
        self.assertEqual([point["position"] for point in curve], [0, 1, 2])
        self.assertEqual([point["relative_position"] for point in curve], [0.0, 0.5, 1.0])

    def test_position_recovery_drops_wrong_length_samples_rather_than_counting_them_as_misses(self):
        # Positional identity is undefined against a different-length draw: position i of a
        # 4-residue sample is not position i of a 3-residue native. Counting it as a miss would
        # understate recovery for exactly the models that pick their own length.
        curve = position_recovery(["ABC", "ABCD"], "ABC")
        self.assertEqual([point["recovery"] for point in curve], [1.0, 1.0, 1.0])

    def test_position_recovery_reports_none_when_nothing_matched_the_length(self):
        curve = position_recovery(["ABCD"], "ABC")
        self.assertEqual([point["recovery"] for point in curve], [None, None, None])

    def test_position_recovery_of_a_single_residue_loop_sits_mid_axis(self):
        # relative_position would divide by zero at length 1; it is pinned to 0.5 so the point
        # lands in the middle bin instead of claiming to be an N-terminal anchor.
        self.assertEqual(position_recovery(["A"], "A")[0]["relative_position"], 0.5)


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
