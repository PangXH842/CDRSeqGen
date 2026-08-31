import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.export_manifest import build_manifest


def _entry():
    regions = {
        "fwr1": "Q",
        "cdr1": "A",
        "fwr2": "W",
        "cdr2": "C",
        "fwr3": "R",
        "cdr3": "DDD",
        "fwr4": "G",
    }
    return {
        "sequence_id": "example",
        "species": "human",
        "split": "test",
        "source": {"dataset": "OAS", "run": 7},
        "heavy": {**regions, "full_aa": "QAWCRDDDG"},
    }


class ExportManifestTests(unittest.TestCase):
    def test_length_free_track_is_written_to_task_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed.json"
            path.write_text(json.dumps([_entry()]))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".metadata.json").write_text(json.dumps({
                "dataset": "OAS",
                "processed_file_sha256": digest,
            }))

            manifest = build_manifest(
                path,
                split="test",
                sample_size=None,
                selection_seed="test",
                benchmark_id="length_free_test",
                track="length_free",
            )

        self.assertEqual(manifest["task"]["track"], "length_free")
        self.assertEqual(len(manifest["records"]), 1)

    def test_unknown_track_is_rejected_before_reading_input(self):
        with self.assertRaisesRegex(ValueError, "unsupported benchmark track"):
            build_manifest(
                Path("does-not-exist.json"),
                split="test",
                sample_size=None,
                selection_seed="test",
                benchmark_id="bad",
                track="unknown",
            )


if __name__ == "__main__":
    unittest.main()
