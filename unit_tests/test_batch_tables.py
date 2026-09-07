import csv
import tempfile
import unittest
from pathlib import Path

from cdrseqgen.visualization.batch_tables import (
    RUN_RE,
    _fmt,
    _common_recovery_policy,
    build_table,
    dataset_alias,
    markdown_table,
    recovery_summary,
    unseen_fraction,
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
                # Overfitting tail: the last row must NOT be the reported one (README: "read
                # best-epoch, not last-epoch"). Without it, idxmax and "take the last row" agree.
                {"epoch": 3, "eval_accuracy": 0.75, "eval_core_accuracy": 0.31,
                 "eval_anchor_accuracy": 0.88, "eval_core_coverage": 0.80},
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
                {"epoch": 3, "eval_accuracy": 0.72, "eval_core_accuracy": 0.05,
                 "eval_anchor_accuracy": 0.75, "eval_core_coverage": 0.10},
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
                {"epoch": 3, "eval_accuracy": 0.65, "eval_core_accuracy": 0.08,
                 "eval_anchor_accuracy": 0.66, "eval_core_coverage": 1.0},
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


class DatasetTagPlumbingTests(unittest.TestCase):
    """A new corpus tag must survive parsing AND the table lookups, or a batch renders empty."""

    def test_run_names_parse_for_every_declared_dataset_tag(self):
        # RUN_RE used to spell out OAS|SD2|DS. Adding OAS4 for the four-donor corpus then made
        # every OAS4 run unparseable: discover_runs returned nothing and a finished batch produced
        # empty tables and dashboards with no error anywhere.
        from cdrseqgen.train.logging import DATASET_TAGS, UNKNOWN_DATASET_TAG
        for tag in {t for _, t in DATASET_TAGS} | {UNKNOWN_DATASET_TAG}:
            with self.subTest(tag=tag):
                match = RUN_RE.match(f"h3_{tag}_LC_scratch")
                self.assertIsNotNone(match, f"h3_{tag}_LC_scratch does not parse")
                self.assertEqual(match.group("ds"), tag)

    def test_unseen_fraction_measures_the_split_it_names(self):
        """It used to count every non-train entry, merging validation AND test on a 3-way corpus."""
        entries = [
            {"heavy": {"cdr3": "AAA"}, "split": "train"},
            {"heavy": {"cdr3": "AAA"}, "split": "validation"},
            {"heavy": {"cdr3": "BBB"}, "split": "validation"},
            {"heavy": {"cdr3": "CCC"}, "split": "test"},
            {"heavy": {"cdr3": "DDD"}, "split": "test"},
        ]
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="validation")
        self.assertEqual((n, unseen), (2, 1))       # not (4, 3)
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="test")
        self.assertEqual((n, unseen), (2, 2))

    def test_unseen_fraction_falls_back_when_there_is_no_such_split(self):
        entries = [{"heavy": {"cdr3": "AAA"}, "split": "train"},
                   {"heavy": {"cdr3": "BBB"}, "split": "test"}]
        _, _, n, unseen = unseen_fraction(entries, "cdr3", split="validation")
        self.assertEqual((n, unseen), (1, 1))

    def test_a_missing_metric_renders_as_a_dash_not_nan(self):
        """`aggregate` returns None for an uncollected metric; the table must not crash or print
        'nan'. Both were live: causal H3 has no per-position entropy, since that needs
        length-matched candidates."""
        self.assertEqual(_fmt(None), "—")
        self.assertEqual(_fmt(0.5), "0.500")

    def test_a_second_oas_corpus_resolves_onto_the_canonical_oas_rows(self):
        # Tables are written against "OAS"; a batch on OAS4 must still fill them.
        self.assertEqual(dataset_alias({("h3", "OAS4", "", "scratch"): "d"}), {"OAS": "OAS4"})
        self.assertEqual(dataset_alias({("h3", "OAS", "", "scratch"): "d"}), {})
        # Ambiguous or absent: leave the canonical tag alone rather than guessing.
        self.assertEqual(dataset_alias({("h3", "OAS", "", "s"): "d",
                                        ("h3", "OAS4", "", "s"): "e"}), {})
        self.assertEqual(dataset_alias({("h3", "SD2", "_LC", "scratch"): "d"}), {})


class MixedLossDetectionTests(unittest.TestCase):
    """A core-weighted run optimises the number the recovery tables report, so a batch that mixes
    objectives is not one comparison — it must say so rather than render silently."""

    def _batch(self, tmp, losses):
        import json, os
        runs = {}
        for i, loss in enumerate(losses):
            run = Path(tmp) / f"h3_OAS_scratch_{i}"
            run.mkdir(parents=True)
            if loss is not None:
                (run / "run_config.json").write_text(json.dumps({"loss_type": loss}))
            runs[("h3", "OAS", "", f"m{i}")] = str(run)
        return runs

    def test_single_objective_is_not_flagged(self):
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(loss_warning(self._batch(tmp, ["cross_entropy", "cross_entropy"])))

    def test_mixed_objectives_are_flagged(self):
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            warning = loss_warning(self._batch(tmp, ["cross_entropy", "weighted_cross_entropy"]))
            self.assertIsNotNone(warning)
            self.assertIn("weighted_cross_entropy", warning)

    def test_runs_predating_the_field_count_as_cross_entropy(self):
        """Old runs have no loss_type; they trained with plain cross-entropy, so a batch of them
        plus a new cross-entropy run is still one comparison."""
        import tempfile
        from cdrseqgen.visualization.batch_tables import loss_warning
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(loss_warning(self._batch(tmp, [None, "cross_entropy"])))


class MethodComparisonTableTests(unittest.TestCase):
    def test_every_registered_method_has_a_display_family(self):
        """A method whose run names parse to a family absent from FAMILY_LABELS is silently
        dropped from every table and dashboard — the run completes, the artifacts are written, and
        the results simply never appear. Adding a method must add its family."""
        from cdrseqgen.methods.registry import METHODS
        from cdrseqgen.visualization.batch_tables import FAMILY_LABELS
        for name in METHODS:
            # `mlm` is the historical default and its runs carry no method marker in the name.
            family = "scratch" if name == "mlm" else f"{name}_scratch"
            self.assertIn(family, FAMILY_LABELS,
                          f"method {name!r} produces runs named *_{family} with no table family")

    def test_family_to_method_mapping_comes_from_the_registry(self):
        """Dashboards branch on this to decide whether teacher-forced bars are comparable. A
        prefix guess (`startswith("ar_")`) filed every method added after `ar` under "masked",
        which inverts the warning: `s2s` predicts position i from the true prefix exactly as `ar`
        does, so it is the MASKED arm that is the odd one out."""
        from cdrseqgen.methods.registry import METHODS
        from cdrseqgen.visualization.batch_tables import FAMILY_LABELS, method_of_family
        self.assertEqual(method_of_family("scratch"), "mlm")
        self.assertEqual(method_of_family("esm2"), "mlm")
        self.assertEqual(method_of_family("ar_scratch"), "ar")
        self.assertEqual(method_of_family("s2s_scratch"), "s2s")
        for family in FAMILY_LABELS:
            self.assertIn(method_of_family(family), METHODS)

    def test_compare_suite_table_contains_masked_and_causal_runs(self):
        row = {
            "epoch": 1,
            "eval_accuracy": 0.5,
            "eval_core_accuracy": 0.4,
            "eval_anchor_accuracy": 0.8,
            "eval_core_coverage": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            masked = Path(tmp) / "h3_OAS_scratch"
            causal = Path(tmp) / "h3_OAS_ar_scratch"
            write_metrics(masked, [row])
            write_metrics(causal, [{**row, "eval_core_accuracy": 0.45}])
            table = build_table("methods", {
                ("h3", "OAS", "", "scratch"): str(masked),
                ("h3", "OAS", "", "ar_scratch"): str(causal),
            })

        self.assertIn("scratch (masked)", table)
        self.assertIn("scratch (causal)", table)
        self.assertIn("0.4000", table)
        self.assertIn("0.4500", table)
        self.assertIn("+0.050", table)

    def test_the_encoder_decoder_arm_reaches_the_comparison_table(self):
        row = {
            "epoch": 1,
            "eval_accuracy": 0.5,
            "eval_core_accuracy": 0.4,
            "eval_anchor_accuracy": 0.8,
            "eval_core_coverage": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            masked = Path(tmp) / "h3_OAS_scratch"
            seq2seq = Path(tmp) / "h3_OAS_s2s_scratch"
            write_metrics(masked, [row])
            write_metrics(seq2seq, [{**row, "eval_core_accuracy": 0.47}])
            table = build_table("methods", {
                ("h3", "OAS", "", "scratch"): str(masked),
                ("h3", "OAS", "", "s2s_scratch"): str(seq2seq),
            })

        self.assertIn("scratch (enc-dec)", table)
        self.assertIn("0.4700", table)


class GenerationPolicyReportingTests(unittest.TestCase):
    def test_length_policy_and_recovery_population_come_from_each_artifact(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            # The same model family can use native or inferred length: never infer policy from
            # the method name. An entirely unmatched framework must not inflate the denominator.
            for family, source, mode, expected in (
                ('scratch', 'length_predictor', 'inferred', 'inferred'),
                ('ar_scratch', 'eos_token', 'inferred', 'EOS'),
                ('esm2', 'native_mask_count', 'fixed', 'native length'),
            ):
                run = Path(directory) / family
                run.mkdir()
                payload = {'length_policy': {'mode': mode, 'source': source},
                           'records': [{'metrics': {'aar_mean': 0.5}}, {'metrics': {'aar_mean': None}}]}
                (run / 'generation_eval.json').write_text(json.dumps(payload))
                runs[('h3', 'OAS', '', family)] = str(run)
                table = build_table('design', runs)
                self.assertIn(f'| {expected} | 1/2 | 0.500 |', table)
            self.assertNotIn('a masked run is given the length', table)
            self.assertIn('populations may differ', table)

    def test_legacy_artifact_without_policy_is_reported_as_unknown(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'generation_eval.json').write_text(json.dumps({
                'records': [{'metrics': {'aar_mean': 0.5}}]}))
            table = build_table('design', {('h3', 'OAS', '', 'scratch'): directory})
        self.assertIn('| unknown | 1/1 | 0.500 |', table)


if __name__ == "__main__":
    unittest.main()
