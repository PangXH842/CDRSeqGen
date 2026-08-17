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
