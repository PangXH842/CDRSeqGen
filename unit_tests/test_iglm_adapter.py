import unittest

from benchmark.adapters.iglm import (
    InvalidGeneration,
    assert_resume_compatible,
    extract_infilled_loop,
    run_example,
    stable_example_seed,
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
        "chain_token": "[HEAVY]",
        "species_token": "[HUMAN]",
    }


class FakeSampler:
    def __init__(self, draws):
        self.draws = iter(draws)
        self.seed = None
        self.prepared = None

    def set_seed(self, seed):
        self.seed = seed

    def prepare(self, record):
        self.prepared = record["example_id"]

    def draw(self, **_kwargs):
        value = next(self.draws)
        if isinstance(value, Exception):
            raise value
        return value


class IgLMAdapterTests(unittest.TestCase):
    def test_extracts_loop_only_when_both_flanks_are_preserved(self):
        self.assertEqual(extract_infilled_loop("QQCCCWW", "QQ", "WW"), "CCC")
        with self.assertRaisesRegex(InvalidGeneration, "pt1"):
            extract_infilled_loop("RRCCCWW", "QQ", "WW")
        with self.assertRaisesRegex(InvalidGeneration, "pt2"):
            extract_infilled_loop("QQCCCVV", "QQ", "WW")

    def test_attempt_accounting_preserves_variable_lengths_and_duplicates(self):
        sampler = FakeSampler(["QQAAAAWW", "QQCCCWW", "QQCCCWW"])
        result = run_example(
            manifest_record(),
            sampler,
            num_samples=2,
            max_attempts=5,
            temperature=1.0,
            top_p=0.9,
            base_seed=42,
            max_errors=3,
        )

        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["valid_attempts"], 3)
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["samples"], ["CCC", "CCC"])
        self.assertEqual([item["loop"] for item in result["raw_generations"]], ["AAAA", "CCC", "CCC"])
        self.assertTrue(result["complete"])
        self.assertEqual(sampler.prepared, manifest_record()["example_id"])

    def test_invalid_draws_are_recorded_and_bounded(self):
        sampler = FakeSampler(["RRCCCWW", "RRCCCWW", "QQCCCWW"])
        result = run_example(
            manifest_record(),
            sampler,
            num_samples=1,
            max_attempts=5,
            temperature=1.0,
            top_p=0.9,
            base_seed=42,
            max_errors=2,
        )

        self.assertEqual(result["attempts"], 2)
        self.assertFalse(result["complete"])
        self.assertEqual(result["invalid_attempts"], 2)
        self.assertEqual(result["failures"][-1]["code"], "max_consecutive_errors")

    def test_natural_policy_retains_variable_lengths_without_filtering(self):
        sampler = FakeSampler(["QQAAAAWW", "QQCCCWW"])
        sampler.length_policy = "natural"
        result = run_example(
            manifest_record(), sampler,
            num_samples=2, max_attempts=2, temperature=1.0, top_p=0.9,
            base_seed=42, max_errors=3,
        )
        self.assertEqual(result["samples"], ["AAAA", "CCC"])
        self.assertEqual(result["length_matches"], 1)
        self.assertEqual(result["length_mismatches"], 1)
        self.assertEqual(result["length_policy"]["mode"], "inferred")

    def test_example_seed_is_stable_and_id_specific(self):
        self.assertEqual(stable_example_seed(42, "a"), stable_example_seed(42, "a"))
        self.assertNotEqual(stable_example_seed(42, "a"), stable_example_seed(42, "b"))

    def test_resume_rejects_changed_sampling_parameters(self):
        expected = {
            "schema_version": "v1",
            "adapter": "iglm",
            "benchmark_id": "b",
            "manifest": {"sha256": "x"},
            "model": {"name": "IgLM"},
            "sampling": {"temperature": 1.0},
        }
        assert_resume_compatible(dict(expected), expected)
        changed = {**expected, "sampling": {"temperature": 0.8}}
        with self.assertRaisesRegex(ValueError, "sampling"):
            assert_resume_compatible(changed, expected)


if __name__ == "__main__":
    unittest.main()
