import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cdrseqgen.artifacts import (
    RUN_CONFIG_FILENAME,
    find_run_config,
    load_run_config,
)
from generate import resolve_config


class RunConfigLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_current_filename_wins_over_the_legacy_one(self):
        (self.dir / "run_meta.json").write_text(json.dumps({"target_region": "l2"}))
        (self.dir / RUN_CONFIG_FILENAME).write_text(json.dumps({"target_region": "h3"}))

        config, path = load_run_config(self.dir)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(path.name, RUN_CONFIG_FILENAME)

    def test_legacy_filename_still_loads(self):
        (self.dir / "run_meta.json").write_text(json.dumps({"target_region": "h3"}))
        self.assertEqual(load_run_config(self.dir)[0]["target_region"], "h3")

    def test_absent_config_is_distinguishable_from_a_corrupt_one(self):
        self.assertIsNone(find_run_config(self.dir))
        with self.assertRaises(FileNotFoundError):
            load_run_config(self.dir)

        (self.dir / RUN_CONFIG_FILENAME).write_text("{not json")
        self.assertIsNotNone(find_run_config(self.dir))
        with self.assertRaises(json.JSONDecodeError):
            load_run_config(self.dir)


class GenerateConfigSourceTests(unittest.TestCase):
    """resolve_config marks CLI-built configs source='cli', and check_compatibility skips both
    context checks for those — so a corrupt run config must not degrade into that path."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.args = SimpleNamespace(
            model_dir=str(self.dir), model_type=None, target_loop=None,
            partner_chain=None, antigen=None, max_len=128,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_a_saved_config_is_trusted_provenance(self):
        (self.dir / RUN_CONFIG_FILENAME).write_text(json.dumps({
            "target_region": "h3", "model_type": "scratch", "include_partner_chain": True,
        }))
        configs = resolve_config(self.args)
        self.assertEqual(configs["source"], "run_config")
        self.assertTrue(configs["include_partner_chain"])

    def test_missing_config_falls_back_to_cli(self):
        configs = resolve_config(self.args)
        self.assertEqual(configs["source"], "cli")

    def test_preloaded_snapshot_is_reused_and_override_warning_is_retained(self):
        saved = {"model_type": "scratch", "include_partner_chain": True}
        self.args.model_type = "other-backbone"
        with patch("cdrseqgen.methods.common.generation.load_run_config") as reload, \
                self.assertLogs("cdrseqgen.methods.common.generation", level="WARNING") as logs:
            configs = resolve_config(self.args, run_config=saved)

        reload.assert_not_called()
        self.assertEqual(saved["model_type"], "scratch")
        self.assertEqual(configs["model_type"], "other-backbone")
        self.assertEqual(configs["source"], "run_config")
        self.assertTrue(configs["include_partner_chain"])
        self.assertIn("Model type mismatch", logs.output[0])

    def test_corrupt_config_raises_rather_than_silently_disabling_context_checks(self):
        (self.dir / RUN_CONFIG_FILENAME).write_text("{truncated")
        with self.assertRaises(json.JSONDecodeError):
            resolve_config(self.args)


if __name__ == "__main__":
    unittest.main()
