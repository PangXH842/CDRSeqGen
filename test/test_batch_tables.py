import csv
import tempfile
import unittest
from pathlib import Path

from cdrseqgen.utils.batch_tables import (
    _common_recovery_policy,
    build_table,
    markdown_table,
    recovery_summary,
)


FIELDS = (
    "epoch",
    "eval_accuracy",
    "eval_core_accuracy",
    "eval_anchor_accuracy",
    "eval_core_coverage",
)


def write_metrics(run_dir, rows, *, include_coverage=True):
    run_dir.mkdir(parents=True)
    fields = FIELDS if include_coverage else FIELDS[:-1]
    with (run_dir / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


class CoverageAwareReportingTests(unittest.TestCase):
    def test_adequate_coverage_selects_core_best_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.90, "eval_core_accuracy": 0.20,
                 "eval_anchor_accuracy": 0.95, "eval_core_coverage": 0.80},
                {"epoch": 2, "eval_accuracy": 0.80, "eval_core_accuracy": 0.40,
                 "eval_anchor_accuracy": 0.90, "eval_core_coverage": 0.80},
            ])
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "core")
        self.assertEqual(summary["selection_metric"], "eval_core_accuracy")
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["primary_value"], 0.40)
        self.assertEqual(summary["whole_accuracy"], 0.80)
        self.assertEqual(summary["core_coverage"], 0.80)

    def test_low_coverage_selects_whole_and_marks_core_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.70, "eval_core_accuracy": 0.99,
                 "eval_anchor_accuracy": 0.70, "eval_core_coverage": 0.10},
                {"epoch": 2, "eval_accuracy": 0.80, "eval_core_accuracy": 0.10,
                 "eval_anchor_accuracy": 0.80, "eval_core_coverage": 0.10},
            ])
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "whole")
        self.assertEqual(summary["selection_metric"], "eval_accuracy")
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["primary_value"], 0.80)
        self.assertIsNone(summary["core_accuracy"])
        self.assertIn("below", summary["reason"])

    def test_missing_coverage_falls_back_to_explicit_whole_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "legacy"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.60, "eval_core_accuracy": 0.90,
                 "eval_anchor_accuracy": 0.60, "eval_core_coverage": 1.0},
                {"epoch": 2, "eval_accuracy": 0.70, "eval_core_accuracy": 0.10,
                 "eval_anchor_accuracy": 0.70, "eval_core_coverage": 1.0},
            ], include_coverage=False)
            summary = recovery_summary(str(run))

        self.assertEqual(summary["primary_metric"], "whole")
        self.assertEqual(summary["epoch"], 2)
        self.assertIsNone(summary["core_coverage"])
        self.assertEqual(summary["reason"], "core coverage unavailable")

    def test_loop_name_does_not_determine_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "l2_OAS_scratch"
            write_metrics(run, [
                {"epoch": 1, "eval_accuracy": 0.30, "eval_core_accuracy": 0.40,
                 "eval_anchor_accuracy": 0.20, "eval_core_coverage": 0.80},
            ])
            table = build_table("loops", {("l2", "OAS", "", "scratch"): str(run)})

        self.assertIn("| L2 | core | 80.0% | 0.4000", table)
        self.assertNotIn("L2 whole-span", table)

    def test_mixed_policies_are_detected_instead_of_subtracted(self):
        core = {"primary_metric": "core", "core_coverage": 1.0}
        whole = {"primary_metric": "whole", "core_coverage": 0.1}
        self.assertEqual(_common_recovery_policy((core, whole)), ("mixed", 0.1))

    def test_markdown_renderer_rejects_wrong_row_width(self):
        with self.assertRaisesRegex(ValueError, "cells"):
            markdown_table("Example", ("a", "b"), (("only one",),))


if __name__ == "__main__":
    unittest.main()
