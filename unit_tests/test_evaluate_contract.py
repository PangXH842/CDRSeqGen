import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cdrseqgen.methods.mlm.eval_cli import resolve_decoding
from evaluate import (
    compatibility_problems,
    default_output_path,
    evaluation_provenance,
)


class EvaluationCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(model_type="scratch")
        self.data = SimpleNamespace(
            target_region="h3",
            include_partner_chain=False,
            include_antigen=False,
            data_path="data/processed/processed_oas_tuning.json",
            max_len=128,
            masking_policy="full_span",
        )
        self.run = {
            "target_region": "h3",
            "model_type": "scratch",
            "include_partner_chain": False,
            "include_antigen": False,
            "data_path": self.data.data_path,
            "max_len": 128,
            "masking_policy": "full_span",
        }

    def test_matching_input_contract_has_no_problems(self):
        self.assertEqual(compatibility_problems(self.run, self.model, self.data), [])

    def test_every_input_shaping_mismatch_is_reported(self):
        mismatched = dict(self.run)
        mismatched.update({
            "target_region": "h1",
            "model_type": "Exscientia/IgBert",
            "include_partner_chain": True,
            "include_antigen": True,
            "data_path": "different.json",
            "max_len": 512,
            "masking_policy": "mixed",
        })
        problems = compatibility_problems(mismatched, self.model, self.data)
        self.assertEqual(len(problems), 7)
        self.assertTrue(any("target CDR" in item for item in problems))
        self.assertTrue(any("backbone/tokenizer" in item for item in problems))

    def test_absent_legacy_fields_are_unknown_not_mismatches(self):
        self.assertEqual(
            compatibility_problems({"target_region": "h3"}, self.model, self.data),
            [],
        )


class DecoderContractTests(unittest.TestCase):
    def test_full_span_ignores_an_old_unsafe_recorded_default(self):
        chosen = resolve_decoding(
            None,
            run_config={"masking_policy": "full_span", "gen_decoding": "iterative"},
        )
        self.assertEqual(chosen, "independent")

    def test_missing_policy_is_treated_as_legacy_full_span(self):
        self.assertEqual(
            resolve_decoding(None, run_config={"gen_decoding": "iterative"}),
            "independent",
        )

    def test_mixed_policy_uses_its_recorded_decoder_or_iterative(self):
        self.assertEqual(
            resolve_decoding(
                None,
                run_config={"masking_policy": "mixed", "gen_decoding": "independent"},
            ),
            "independent",
        )
        self.assertEqual(
            resolve_decoding(None, run_config={"masking_policy": "mixed"}),
            "iterative",
        )

    def test_explicit_off_distribution_request_remains_possible(self):
        self.assertEqual(
            resolve_decoding(
                "iterative",
                run_config={"masking_policy": "full_span", "gen_decoding": "independent"},
            ),
            "iterative",
        )


class EvaluationArtifactTests(unittest.TestCase):
    def test_default_output_is_parameter_addressed(self):
        args = SimpleNamespace(
            model_dir="/tmp/checkpoint",
            length_mode="inferred",
            min_length=None,
            max_length=None,
            length_temperature=1.0,
            decoding="independent",
            num_examples=25,
            stratify=True,
            num_samples=20,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            seed=42,
            skip_perplexity=False,
            judge="Exscientia/IgBert",
            judge_path=None,
        )
        path = default_output_path(args)
        self.assertIn(os.path.join("checkpoint", "evaluations"), path)
        self.assertIn("inferred_independent_n25x20", path)
        self.assertIn("IgBert", path)

    @patch("cdrseqgen.methods.common.evaluation.git_state", return_value={
        "repository_commit": "abc", "repository_dirty": False,
    })
    def test_provenance_hashes_config_checkpoint_and_metadata(self, _git):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment.json"
            config.write_text("{}")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "run_config.json").write_text("{}")
            weights = checkpoint / "model.safetensors"
            weights.write_bytes(b"weights")
            data = root / "processed.json"
            data.write_text("[]")
            metadata = root / "processed.metadata.json"
            metadata.write_text(json.dumps({"processed_file_sha256": "declared"}))

            provenance = evaluation_provenance(config, checkpoint, data)

        _git.assert_called_once_with(Path(__file__).resolve().parents[1])
        self.assertEqual(
            provenance["checkpoint_weights"][0]["sha256"],
            hashlib.sha256(b"weights").hexdigest(),
        )
        self.assertEqual(provenance["dataset"]["declared_processed_file_sha256"], "declared")
        self.assertEqual(provenance["repository_commit"], "abc")


if __name__ == "__main__":
    unittest.main()
