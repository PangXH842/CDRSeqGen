import json
import tempfile
import unittest
from pathlib import Path

from fit_mlm_length import resolve_config_path


class ResolveMLMLengthConfigTests(unittest.TestCase):
    def test_discovers_canonical_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            config = checkpoint / "run_config.json"
            config.write_text(json.dumps({"model_type": "scratch"}))

            self.assertEqual(resolve_config_path(str(checkpoint)), config.resolve())

    def test_explicit_config_overrides_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "run_config.json").write_text("{}")
            explicit = root / "training.json"
            explicit.write_text("{}")

            self.assertEqual(
                resolve_config_path(str(checkpoint), str(explicit)), explicit.resolve()
            )

    def test_missing_config_fails_with_actionable_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "pass --config explicitly"):
                resolve_config_path(directory)


if __name__ == "__main__":
    unittest.main()
