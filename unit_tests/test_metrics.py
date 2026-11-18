"""Tests merged from: test_metrics.py, test_blosum.py, test_confidence_metrics.py.

Each section below was a test file of its own; the original module docstring is kept as a comment
at the head of its section. Names that more than one file used for different helpers were renamed
with the file name as a prefix.
"""

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
from cdrseqgen.eval.metrics import (
    BLOSUM62,
    BLOSUM62_UNKNOWN,
    amino_acid_recovery,
    blosum_aligned,
    blosum_aligned_ratio,
    blosum_identity_ratio,
    blosum_pair,
    blosum_positional,
    blosum_provenance,
    calculate,
)
from cdrseqgen.eval.metrics import confidence_metrics, spearman


# ====================================================================================================
# from test_metrics.py
# ====================================================================================================

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


# ====================================================================================================
# from test_blosum.py
# BLOSUM62 scoring: the matrix, both score modes, and the boundaries between them.
#
# The point of the metric is that exact-match recovery scores I->V and I->P identically (zero), so
# every test here pins a case where BLOSUM and recovery must disagree.
# ====================================================================================================

NATIVE = "ARDITRRDCSGGSCYSFPYYYYYMDV"

class MatrixTests(unittest.TestCase):
    def test_the_table_is_complete_and_symmetric(self):
        self.assertEqual(len(BLOSUM62), 400)
        for (a, b), value in BLOSUM62.items():
            self.assertEqual(BLOSUM62[(b, a)], value, f"asymmetric at {a}/{b}")

    def test_known_entries(self):
        for a, b, expected in [("A", "A", 4), ("W", "W", 11), ("C", "C", 9),
                               ("I", "V", 3), ("K", "R", 2), ("Y", "F", 3), ("I", "P", -3)]:
            with self.subTest(pair=f"{a}{b}"):
                self.assertEqual(blosum_pair(a, b), expected)

    def test_an_unknown_residue_scores_as_a_mismatch_not_a_skip(self):
        # Skipping it would shorten the denominator and flatter the sequence containing it.
        self.assertEqual(blosum_pair("X", "A"), BLOSUM62_UNKNOWN)
        self.assertEqual(blosum_pair("-", "-"), BLOSUM62_UNKNOWN)


class PositionalTests(unittest.TestCase):
    def test_identical_scores_one(self):
        self.assertAlmostEqual(blosum_identity_ratio(NATIVE, NATIVE), 1.0)

    def test_a_conservative_substitution_costs_recovery_but_not_blosum(self):
        conservative = NATIVE.replace("I", "V")          # I->V is +3, near I->I's +4
        self.assertLess(amino_acid_recovery(conservative, NATIVE), 1.0)
        self.assertGreater(blosum_identity_ratio(conservative, NATIVE), 0.98)

    def test_a_radical_substitution_is_penalised_far_harder_than_recovery_shows(self):
        radical = NATIVE.replace("Y", "P").replace("D", "W")
        self.assertGreater(amino_acid_recovery(radical, NATIVE), 0.6)
        self.assertLess(blosum_identity_ratio(radical, NATIVE), 0.5)

    def test_unequal_lengths_have_no_positional_score(self):
        # No positional correspondence exists; the aligned score is what covers this case.
        self.assertIsNone(blosum_positional(NATIVE[:-3], NATIVE))
        self.assertIsNone(blosum_identity_ratio(NATIVE + "GGG", NATIVE))

    def test_an_empty_index_subset_is_none_not_zero(self):
        self.assertIsNone(blosum_positional(NATIVE, NATIVE, indices=[]))


class AlignedTests(unittest.TestCase):
    def test_identical_scores_one(self):
        self.assertAlmostEqual(blosum_aligned_ratio(NATIVE, NATIVE), 1.0)

    def test_it_is_defined_for_unequal_lengths(self):
        for other in (NATIVE[:-3], NATIVE + "GGG", NATIVE[5:]):
            with self.subTest(other=other):
                self.assertIsNotNone(blosum_aligned_ratio(other, NATIVE))

    def test_deleting_residues_costs_score(self):
        self.assertLess(blosum_aligned(NATIVE[:-3], NATIVE), blosum_aligned(NATIVE, NATIVE))

    def test_it_charges_for_extra_residues_that_recovery_ignores(self):
        # amino_acid_recovery divides by len(native), so a longer sequence containing the whole
        # native still scores 1.0. The alignment does not let that pass.
        longer = NATIVE + "GGG"
        self.assertEqual(amino_acid_recovery(longer, NATIVE), 1.0)
        self.assertLess(blosum_aligned_ratio(longer, NATIVE), 1.0)

    def test_an_unrelated_sequence_scores_below_zero(self):
        self.assertLess(blosum_aligned_ratio("M" * len(NATIVE), NATIVE), 0.0)

    def test_adjacent_opposing_gaps_are_excluded(self):
        # D/W scores -4. Opposing one-residue gaps would score -2 with these
        # penalties, but alignment version 1 excludes that path.
        self.assertEqual(blosum_aligned("D", "W", gap_open=-1, gap_extend=-1), -4.0)


class ReportingTests(unittest.TestCase):
    def test_calculate_reports_both_views_and_they_differ_in_population(self):
        samples = [NATIVE, NATIVE.replace("I", "V"), NATIVE[:-3], "M" * 20]
        out = calculate(samples, native=NATIVE)
        # @L covers the 2 length-matched samples; aligned covers all 4.
        self.assertIn("blosum62_ratio_at_length_mean", out)
        self.assertIn("blosum62_aligned_ratio_mean", out)
        self.assertGreater(out["blosum62_ratio_at_length_mean"],
                           out["blosum62_aligned_ratio_mean"])

    def test_no_native_means_no_blosum_at_all(self):
        out = calculate(["ACDE", "ACDF"], native=None)
        self.assertFalse([k for k in out if "blosum" in k])

    def test_aligned_means_retain_duplicates_and_exclude_empty_draws(self):
        out = calculate(["AC", "AC", "AQ", "ACD", ""], native="AC")
        # Native self-score is 13; candidate scores are 13, 13, 1, 2, None.
        self.assertEqual(out["blosum62_aligned_mean"], 29 / 4)
        self.assertEqual(out["blosum62_aligned_ratio_mean"], (1 + 1 + 1/13 + 2/13) / 4)
        self.assertEqual(out["n_samples"], 5)
        self.assertEqual(out["length_match_rate"], 3 / 5)

    def test_empty_candidates_have_no_aligned_scores(self):
        out = calculate(["", ""], native="AC")
        self.assertNotIn("blosum62_aligned_mean", out)
        self.assertNotIn("blosum62_aligned_ratio_mean", out)
        self.assertEqual(out["n_samples"], 2)

    def test_zero_native_self_score_omits_ratios_without_losing_recovery(self):
        # A/A=4 and X/X=-4 give a zero positional and aligned denominator.
        out = calculate(["AX"], native="AX")
        self.assertNotIn("blosum62_ratio_at_length_mean", out)
        self.assertNotIn("blosum62_aligned_ratio_mean", out)
        self.assertEqual(out["blosum62_mean_at_length"], 0.0)
        self.assertEqual(out["aar_mean"], 1.0)

    def test_provenance_names_everything_a_score_depends_on(self):
        p = blosum_provenance()
        for key in ("matrix", "gap_open", "gap_extend", "alignment",
                    "positional_normalisation", "aligned_normalisation"):
            self.assertIn(key, p)
        self.assertEqual(p["matrix"], "BLOSUM62")
        self.assertEqual(p["alignment_version"], 1)
        self.assertFalse(p["adjacent_opposing_gaps"])
        self.assertEqual(p["gap_cost_formula"], "gap_open + (length - 1) * gap_extend")


# ====================================================================================================
# from test_confidence_metrics.py
# ====================================================================================================

class SpearmanTests(unittest.TestCase):
    def test_perfect_and_inverted_rank_agreement(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_undefined_cases_return_none_rather_than_zero(self):
        """A constant score ranks nothing — reporting 0.0 would imply 'measured, no effect'."""
        self.assertIsNone(spearman([1, 2], [1, 2]))          # n < 3
        self.assertIsNone(spearman([5, 5, 5, 5], [1, 2, 3, 4]))  # no variance in x
        self.assertIsNone(spearman([1, 2, 3, 4], [7, 7, 7, 7]))  # no variance in y

    def test_ties_take_averaged_ranks(self):
        self.assertAlmostEqual(spearman([1, 2, 2, 3], [1, 2, 2, 3]), 1.0)


class ConfidenceMetricTests(unittest.TestCase):
    """Perplexity is lower-is-better, so orientation is the thing most likely to be wrong."""

    def test_perplexity_is_oriented_low_equals_confident(self):
        recoveries = [0.9, 0.5, 0.1]
        perplexities = [1.0, 5.0, 9.0]          # best candidate has the LOWEST perplexity
        out = confidence_metrics(recoveries, perplexities, prefix="c")
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_higher_is_better_scores_are_supported(self):
        out = confidence_metrics([0.9, 0.5, 0.1], [1.0, 0.5, 0.0],
                                 lower_is_better=False, prefix="c")
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_baseline_is_the_mean_over_the_same_pairs(self):
        """Lift must be measured against the same population, not against aar_mean."""
        out = confidence_metrics([1.0, 0.0, 0.5], [1.0, 2.0, 3.0], prefix="c")
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.5)
        self.assertAlmostEqual(out["c_top1_recovery"], 1.0)

    def test_missing_values_drop_the_pair_not_the_alignment(self):
        # candidate 2's judge score is missing; it must not shift the others' pairing
        out = confidence_metrics([0.9, 0.4, 0.1, 0.8], [1.0, None, 9.0, 2.0], prefix="c")
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_too_few_pairs_reports_n_only(self):
        out = confidence_metrics([0.5, 0.4], [1.0, 2.0], prefix="c")
        self.assertEqual(out, {"c_n": 2})

    def test_unmatched_length_candidates_are_excluded_by_none_recovery(self):
        out = confidence_metrics([None, 0.5, 0.9, 0.2], [1.0, 2.0, 3.0, 4.0], prefix="c")
        self.assertEqual(out["c_n"], 3)

    def test_unequal_lengths_raise_before_dropping_missing_values(self):
        for recoveries, scores in (([0.9, 0.5, 0.1], [1.0, 2.0]),
                                   ([0.9, 0.5], [1.0, 2.0, 3.0]),
                                   ([], [None])):
            with self.subTest(recoveries=recoveries, scores=scores):
                with self.assertRaisesRegex(ValueError, "equal-length"):
                    confidence_metrics(recoveries, scores)

    def test_nonfinite_scores_drop_pairs_without_shifting_candidates(self):
        out = confidence_metrics(
            [0.9, 0.4, 0.1, 0.8, 0.6, 0.7],
            [1.0, float("nan"), 9.0, 2.0, float("inf"), float("-inf")],
            prefix="c",
        )
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.6)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)
        self.assertAlmostEqual(out["c_top3_recovery"], 0.6)

    def test_nonfinite_recoveries_are_excluded(self):
        out = confidence_metrics(
            [0.9, float("nan"), 0.5, float("inf"), 0.1, float("-inf")],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            prefix="c",
        )
        self.assertEqual(out["c_n"], 3)
        self.assertAlmostEqual(out["c_spearman"], 1.0)
        self.assertAlmostEqual(out["c_baseline_recovery"], 0.5)
        self.assertAlmostEqual(out["c_top1_recovery"], 0.9)

    def test_reporting_minimum_uses_only_finite_pairs(self):
        out = confidence_metrics([0.5, 0.4, 0.9], [1.0, 2.0, float("nan")], prefix="c")
        self.assertEqual(out, {"c_n": 2})

    def test_no_eligible_pairs_reports_zero(self):
        out = confidence_metrics(
            [None, float("nan"), float("inf"), float("-inf"), 0.5, 0.6, 0.7],
            [1.0, 1.0, 1.0, 1.0, None, float("nan"), float("inf")],
            prefix="c",
        )
        self.assertEqual(out, {"c_n": 0})


if __name__ == "__main__":
    unittest.main()
