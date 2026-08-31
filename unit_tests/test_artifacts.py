import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from cdrseqgen.artifacts import atomic_write_json, file_identity
from cdrseqgen.train.logging import write_run_config


class AtomicArtifactTests(unittest.TestCase):
    def test_atomic_writer_creates_parents_and_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "artifact.json"
            atomic_write_json(output, {"value": 3})
            self.assertEqual(json.loads(output.read_text()), {"value": 3})
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_serialization_failure_preserves_the_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            output.write_text('{"old": true}\n')
            with self.assertRaises(TypeError):
                atomic_write_json(output, {"not_json": {1, 2, 3}})
            self.assertEqual(json.loads(output.read_text()), {"old": True})
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_file_identity_records_size_and_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"abc")
            identity = file_identity(path)
        self.assertEqual(identity["size_bytes"], 3)
        self.assertEqual(identity["sha256"], hashlib.sha256(b"abc").hexdigest())


@dataclass
class ModelArgs:
    model_type: str = "scratch"


@dataclass
class DataArgs:
    data_path: str
    target_region: str = "h3"


@dataclass
class GenerationArgs:
    gen_decoding: str = "independent"


class TrainingArgs:
    def to_dict(self):
        return {"seed": 42}


class RunConfigProvenanceTests(unittest.TestCase):
    @patch("cdrseqgen.train.logging._package_versions", return_value={"torch": "test"})
    @patch("cdrseqgen.train.logging.git_state", return_value={
        "repository_commit": "abc", "repository_dirty": False,
    })
    def test_run_config_carries_dataset_runtime_and_repository_identity(self, _git, _versions):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "processed.json"
            data.write_text("[]")
            metadata = root / "processed.metadata.json"
            metadata.write_text(json.dumps({"processed_file_sha256": "declared"}))

            payload = write_run_config(
                str(root),
                ModelArgs(),
                DataArgs(str(data)),
                GenerationArgs(),
                TrainingArgs(),
            )
            written = json.loads((root / "run_config.json").read_text())

        self.assertEqual(written, payload)
        self.assertEqual(payload["provenance"]["dataset"]["declared_processed_file_sha256"], "declared")
        self.assertEqual(payload["provenance"]["runtime"]["packages"]["torch"], "test")
        self.assertEqual(payload["provenance"]["repository_commit"], "abc")


if __name__ == "__main__":
    unittest.main()
