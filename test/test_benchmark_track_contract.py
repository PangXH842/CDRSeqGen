import unittest

from benchmark.scorer import ADAPTER_SCHEMA_VERSION, validate_payload


class BenchmarkTrackContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "benchmark_id": "b",
            "task": {"track": "length_free"},
            "records": [{"example_id": "x"}],
        }
        self.payload = {
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "benchmark_id": "b",
            "manifest": {"sha256": "hash"},
            "sampling": {"track": "length_free"},
            "records": [{"example_id": "x"}],
        }

    def test_matching_track_passes(self):
        ids = validate_payload(
            self.payload,
            label="model",
            manifest=self.manifest,
            manifest_hash="hash",
            allow_partial=False,
        )
        self.assertEqual(ids, ["x"])

    def test_examples_that_generated_nothing_are_refused(self):
        """attempts == 0 is self-consistent, so only this check separates a half-dead run from
        a complete one — every downstream metric silently skips those examples."""
        manifest = {
            "benchmark_id": "b",
            "task": {"track": "length_free"},
            "records": [{"example_id": name} for name in ("x", "y", "z", "w")],
        }
        records = [
            {"example_id": "x", "attempts": 0},
            {"example_id": "y", "attempts": 0},
            {"example_id": "z", "attempts": 20},
            {"example_id": "w", "attempts": 20},
        ]
        payload = {**self.payload, "records": records}
        with self.assertRaisesRegex(ValueError, "zero generation attempts"):
            validate_payload(payload, label="model", manifest=manifest,
                             manifest_hash="hash", allow_partial=False)

        ids = validate_payload(payload, label="model", manifest=manifest,
                               manifest_hash="hash", allow_partial=False,
                               allow_dead_records=True)
        self.assertEqual(ids, ["x", "y", "z", "w"])

    def test_mismatched_track_is_rejected(self):
        payload = {**self.payload, "sampling": {"track": "fixed_native_length"}}
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            validate_payload(
                payload,
                label="model",
                manifest=self.manifest,
                manifest_hash="hash",
                allow_partial=False,
            )


if __name__ == "__main__":
    unittest.main()
