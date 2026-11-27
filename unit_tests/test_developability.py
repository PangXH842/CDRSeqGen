"""Unit tests for benchmark/analysis/developability.py (pure-Python CDR-H3 liability profiler)."""

import unittest

from benchmark.analysis.developability import (
    family_name,
    group_families,
    score_sequence,
    summarize_records,
)


class DevelopabilityTests(unittest.TestCase):
    def test_clean_loop_has_no_liabilities(self):
        # Clean sequence: no N[^P][ST], no C, no NG/NS/NT, no DG/DS, no M/W, no hydro run >= 5.
        m = score_sequence("ARDFGYKR")
        self.assertEqual(m["major_liability_free_rate"], 1.0)
        self.assertEqual(m["nglyc_rate"], 0.0)
        self.assertEqual(m["unpaired_cys_rate"], 0.0)
        self.assertEqual(m["deamidation_ng_rate"], 0.0)
        self.assertEqual(m["isomerization_dg_rate"], 0.0)
        self.assertEqual(m["oxidation_mw_rate"], 0.0)
        self.assertEqual(m["hydrophobic_patch5_rate"], 0.0)
        self.assertAlmostEqual(m["net_charge_ph74_mean"], 2.0)  # R + K + R - D = 3 - 1 = 2
        self.assertAlmostEqual(m["aromatic_fraction_mean"], 2.0 / 8.0)

    def test_nglyc_respects_proline_exclusion(self):
        self.assertEqual(score_sequence("ARNPSDY")["nglyc_rate"], 0.0)  # N-P-S is NOT N-glyc
        self.assertEqual(score_sequence("ARNASDY")["nglyc_rate"], 1.0)
        self.assertEqual(score_sequence("ARNGTDY")["nglyc_rate"], 1.0)
        self.assertEqual(score_sequence("ARNASDY")["major_liability_free_rate"], 0.0)

    def test_unpaired_vs_paired_cysteines(self):
        self.assertEqual(score_sequence("ARDCY")["unpaired_cys_rate"], 1.0)
        self.assertEqual(score_sequence("ARDCY")["major_liability_free_rate"], 0.0)
        self.assertEqual(score_sequence("ARDCCY")["unpaired_cys_rate"], 0.0)
        self.assertEqual(score_sequence("ARDCCY")["major_liability_free_rate"], 1.0)

    def test_deamidation_isomerization_oxidation_and_hydrophobic_patch(self):
        m = score_sequence("ARNGDGMVVVVV")
        self.assertEqual(m["deamidation_ng_rate"], 1.0)
        self.assertEqual(m["isomerization_dg_rate"], 1.0)
        self.assertEqual(m["oxidation_mw_rate"], 1.0)
        self.assertEqual(m["hydrophobic_patch5_rate"], 1.0)
        self.assertEqual(m["major_liability_free_rate"], 0.0)

    def test_macro_framework_averaging_and_family_grouping(self):
        recs = [
            {"samples": ["ARDFGYKR", "ARDCY"]},  # fw1: 50% major-free
            {"samples": ["ARDFGYKR", "ARDFGYKR"]},  # fw2: 100% major-free
        ]
        s = summarize_records(recs)
        self.assertEqual(s["frameworks"], 2)
        self.assertEqual(s["candidates"], 4)
        self.assertAlmostEqual(s["metrics"]["major_liability_free_rate"]["macro_mean"], 0.75)

        grouped = group_families({"mlm_s42": s, "mlm_s43": s, "iglm": s})
        self.assertEqual(family_name("mlm_s42"), "mlm_scratch")
        self.assertEqual(grouped["mlm_scratch"]["seeds"], 2)
        self.assertEqual(grouped["iglm"]["seeds"], 1)


if __name__ == "__main__":
    unittest.main()
