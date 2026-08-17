import tempfile
import unittest
from pathlib import Path

from benchmark.plot_comparison import (
    SCORE_SCHEMA_VERSION,
    benchmark_track,
    build_figure,
    is_development_artifact,
    normalized_position_curve,
    save_comparison_figure,
    validate_scores,
)


def metric(value, examples=2):
    return {"macro_mean": value, "between_example_sd": 0.0, "examples": examples}


def scored_run(label, recovery, diversity):
    aggregate = {
        "whole_span_aar_mean": metric(recovery + 0.1),
        "core_aar_mean": metric(recovery),
        "anchor_aar_mean": metric(recovery + 0.3),
        "diversity_hamming": metric(diversity),
    }
    by_length = {
        name: {"core_aar_mean": metric(recovery, examples=count)}
        for name, count in (("short", 1), ("medium", 1), ("long", 0))
    }
    records = [
        {
            "metrics": {
                "position_recovery": [
                    {"relative_position": 0.0, "recovery": 0.0},
                    {"relative_position": 0.04, "recovery": 1.0},
                    {"relative_position": 1.0, "recovery": recovery},
                ]
            }
        },
        {
            "metrics": {
                "position_recovery": [
                    {"relative_position": 0.0, "recovery": 1.0},
                    {"relative_position": 1.0, "recovery": recovery},
                ]
            }
        },
    ]
    return {
        "label": label,
        "adapter": label,
        "sampling": {"track": "fixed_native_length", "num_samples": 5},
        "aggregate": aggregate,
        "by_length_bucket": by_length,
        "records": records,
    }


def score_payload():
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "manifest": {"benchmark_id": "validation_smoke"},
        "runs": [
            scored_run("cdrseqgen", recovery=0.4, diversity=0.5),
            scored_run("iglm", recovery=0.2, diversity=0.8),
        ],
    }


def length_free_payload():
    payload = score_payload()
    payload["manifest"]["benchmark_id"] = "validation_length_free"
    for index, run in enumerate(payload["runs"]):
        run["sampling"]["track"] = "length_free"
        run["aggregate"].update({
            "length_match_rate": metric(0.6 - index * 0.1),
            "length_absolute_error_mean": metric(0.8 + index * 0.2),
            "edit_similarity_mean": metric(0.9 - index * 0.1),
            "valid_generation_rate": metric(1.0),
            "micro_accounting": {"examples": 2},
        })
        for summary in run["by_length_bucket"].values():
            summary["length_absolute_error_mean"] = metric(1.0)
    # The length-free dashboard must remain valid even if one model has no exact-length draws.
    payload["runs"][1]["aggregate"].update({
        "whole_span_aar_mean": metric(None, 0),
        "core_aar_mean": metric(None, 0),
        "anchor_aar_mean": metric(None, 0),
    })
    return payload


class BenchmarkPlotterTests(unittest.TestCase):
    def test_position_curve_macro_averages_within_each_framework_first(self):
        _, recovery, counts = normalized_position_curve(score_payload()["runs"][0], bins=11)

        self.assertAlmostEqual(recovery[0], 0.75)
        self.assertEqual(counts[0], 2)
        self.assertAlmostEqual(recovery[-1], 0.4)

    def test_rejects_wrong_schema(self):
        payload = score_payload()
        payload["schema_version"] = "wrong"

        with self.assertRaisesRegex(ValueError, "expected score schema"):
            validate_scores(payload)

    def test_writes_a_nonempty_development_figure(self):
        payload = score_payload()
        self.assertTrue(is_development_artifact(payload))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            written = save_comparison_figure(payload, output)

            self.assertEqual(written, output)
            self.assertGreater(output.stat().st_size, 1000)

    def test_length_free_track_uses_length_dashboard_without_requiring_aar(self):
        payload = length_free_payload()

        self.assertEqual(benchmark_track(payload), "length_free")
        validate_scores(payload)
        figure = build_figure(payload)
        try:
            self.assertEqual(
                figure._suptitle.get_text(), "CDR-H3 length-free benchmark comparison"
            )
            self.assertIn("Length-free", figure.axes[0].get_title())
            self.assertIn("Absolute length error", figure.axes[1].get_title())
        finally:
            import matplotlib.pyplot as plt

            plt.close(figure)

    def test_rejects_mixed_tracks(self):
        payload = score_payload()
        payload["runs"][1]["sampling"]["track"] = "length_free"

        with self.assertRaisesRegex(ValueError, "do not share one benchmark track"):
            validate_scores(payload)


if __name__ == "__main__":
    unittest.main()
