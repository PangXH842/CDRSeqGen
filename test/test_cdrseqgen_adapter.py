import json
import tempfile
import unittest
from pathlib import Path

from benchmark.adapters.cdrseqgen import (
    assert_resume_compatible,
    load_run_config,
    run_example,
    stable_example_seed,
    validate_checkpoint_contract,
)


def manifest_record():
    return {
        "example_id": "oas:7:example",
        "parent_sequence": "QQAAAWW",
        "pt1": "QQ",
        "native": "AAA",
        "pt2": "WW",
        "native_length": 3,
        "infill_range": [2, 5],
    }


class FakeSampler:
    def __init__(self, samples):
        self.samples = samples
        self.prepared = None
        self.received_seed = None

    def prepare(self, record):
        self.prepared = record["example_id"]

    def draw_many(self, **kwargs):
        self.received_seed = kwargs["seed"]
        return list(self.samples)


class CDRSeqGenAdapterTests(unittest.TestCase):
    def test_load_run_config_accepts_legacy_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            legacy = checkpoint / "run_meta.json"
            legacy.write_text(json.dumps({"target_region": "h3"}))

            config, source = load_run_config(checkpoint)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(source, legacy)

    def test_load_run_config_prefers_current_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            legacy = checkpoint / "run_meta.json"
            current = checkpoint / "run_config.json"
            legacy.write_text(json.dumps({"target_region": "l2"}))
            current.write_text(json.dumps({"target_region": "h3"}))

            config, source = load_run_config(checkpoint)

        self.assertEqual(config["target_region"], "h3")
        self.assertEqual(source, current)

    def test_records_samples_and_preserves_duplicates(self):
        sampler = FakeSampler(["CCC", "CCC", "DDD"])
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="iterative", temperature=1.0,
            top_k=0, top_p=0.9, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CCC", "CCC", "DDD"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["length_matches"], 3)
        self.assertEqual(
            result["length_policy"],
            {
                "mode": "fixed",
                "source": "native_mask_count",
                "minimum": 3,
                "maximum": 3,
                "selected_length": 3,
            },
        )

        self.assertTrue(result["complete"])
        self.assertEqual(sampler.prepared, manifest_record()["example_id"])
        self.assertEqual(
            sampler.received_seed,
            stable_example_seed(42, manifest_record()["example_id"]),
        )

    def test_invalid_output_is_audited_not_scored(self):
        sampler = FakeSampler(["CC", "C?C", "CCC"])
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="independent", temperature=1.0,
            top_k=0, top_p=1.0, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CCC"])
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["valid_attempts"], 2)
        self.assertEqual(result["invalid_attempts"], 1)
        self.assertFalse(result["complete"])
        self.assertEqual({f["code"] for f in result["failures"]}, {"length_mismatch", "invalid_amino_acid"})

    def test_length_free_track_retains_all_valid_lengths(self):
        sampler = FakeSampler(["CC", "CCC", "CCCC"])
        sampler.track = "length_free"
        sampler.length_metadata = {"mode": "inferred", "selected_lengths": [2, 3, 4]}
        result = run_example(
            manifest_record(), sampler,
            num_samples=3, decoding="independent", temperature=1.0,
            top_k=0, top_p=1.0, base_seed=42,
        )
        self.assertEqual(result["samples"], ["CC", "CCC", "CCCC"])
        self.assertEqual(result["length_matches"], 1)
        self.assertEqual(result["length_mismatches"], 2)
        self.assertFalse(any(failure["code"] == "length_mismatch" for failure in result["failures"]))
        self.assertEqual(result["length_policy"]["mode"], "inferred")

    def test_checkpoint_contract_rejects_context_or_dataset_mismatch(self):
        manifest = {
            "task": {
                "target_region": "h3",
                "track": "fixed_native_length",
                "partner_chain_context": False,
                "antigen_context": False,
            },
            "source": {"processed_file": "processed.json"},
        }
        config = {
            "target_region": "h3",
            "include_partner_chain": False,
            "include_antigen": False,
            "data_path": "data/processed.json",
        }
        validate_checkpoint_contract(config, manifest)
        with self.assertRaisesRegex(ValueError, "include_partner_chain"):
            validate_checkpoint_contract({**config, "include_partner_chain": True}, manifest)
        with self.assertRaisesRegex(ValueError, "data file"):
            validate_checkpoint_contract({**config, "data_path": "other.json"}, manifest)

    def test_a_failing_example_is_recorded_rather_than_aborting_the_run(self):
        class ExplodingSampler(FakeSampler):
            def prepare(self, record):
                raise ValueError("assembled length 700 exceeds checkpoint max_len 512")

        result = run_example(
            manifest_record(), ExplodingSampler([]),
            num_samples=3, decoding="iterative", temperature=1.0,
            top_k=0, top_p=0.9, base_seed=42,
        )
        self.assertTrue(result["generation_failed"])
        self.assertEqual(result["samples"], [])
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["failures"][0]["code"], "generation_exception")
        self.assertEqual(result["length_policy"]["source"], "generation_failed")

    def test_a_failed_example_does_not_inherit_the_previous_length_policy(self):
        """length_metadata is sampler state; a failure must not report the last example's."""
        class StatefulSampler(FakeSampler):
            def __init__(self, samples):
                super().__init__(samples)
                self.calls = 0

            def prepare(self, record):
                self.length_metadata = None
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("CUDA out of memory")
                self.length_metadata = {"mode": "inferred", "selected_length": 3}

        sampler = StatefulSampler(["CCC"])
        kwargs = dict(num_samples=1, decoding="iterative", temperature=1.0,
                      top_k=0, top_p=0.9, base_seed=42)
        first = run_example(manifest_record(), sampler, **kwargs)
        second = run_example(manifest_record(), sampler, **kwargs)

        self.assertEqual(first["length_policy"]["mode"], "inferred")
        self.assertEqual(second["length_policy"]["mode"], "unavailable")

    def test_resume_rejects_changed_checkpoint_or_sampling(self):
        expected = {
            "schema_version": "v1", "adapter": "cdrseqgen", "benchmark_id": "b",
            "manifest": {"sha256": "m"}, "model": {"checkpoint_sha256": "c"},
            "sampling": {"decoding": "iterative"},
        }
        assert_resume_compatible(dict(expected), expected)
        with self.assertRaisesRegex(ValueError, "model"):
            assert_resume_compatible({**expected, "model": {"checkpoint_sha256": "x"}}, expected)


if __name__ == "__main__":
    unittest.main()
