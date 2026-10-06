"""BLOSUM62 scoring: the matrix, both score modes, and the boundaries between them.

The point of the metric is that exact-match recovery scores I->V and I->P identically (zero), so
every test here pins a case where BLOSUM and recovery must disagree.
"""
import unittest

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


if __name__ == "__main__":
    unittest.main()
