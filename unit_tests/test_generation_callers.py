import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import generate
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm import gen_cli as mlm_gen_cli
from cdrseqgen.methods.mlm.evaluation import run_generative_eval


class FakeModel:
    def eval(self):
        return self


class SharedGenerationCallerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_generate_uses_shared_decoder_without_changing_fixed_length(self, shared_decode):
        shared_decode.return_value = [
            self.tokenizer.encode("WY"),
            self.tokenizer.encode("WW"),
        ]
        args = SimpleNamespace(
            partner_chain=None,
            antigen=None,
            max_len=20,
            decoding="independent",
            num_samples=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            skip_perplexity=True,
            show_samples=0,
            seed=42,
        )

        record = mlm_gen_cli.generate_case(
            FakeModel(),
            self.tokenizer,
            torch.device("cpu"),
            args,
            torch.Generator().manual_seed(42),
            {"name": "fixed", "pt1": "AC", "pt2": "DE", "target_truth": "WY", "min_length": 2, "max_length": 2},
            None,
            generate._print_case,
        )

        self.assertEqual(record["target_length"], 2)
        self.assertEqual(record["length_policy"]["source"], "explicit_bounds")
        self.assertEqual([sample["seq"] for sample in record["samples"]], ["WY", "WW"])
        self.assertEqual(shared_decode.call_args.kwargs["decoding"], "independent")
        self.assertEqual(len(shared_decode.call_args.args[5]), 2)

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_held_out_eval_uses_shared_decoder_and_native_mask_count(self, shared_decode):
        native_ids = self.tokenizer.encode("WY")
        shared_decode.return_value = [native_ids, self.tokenizer.encode("WW")]
        dataset = [{
            "input_ids": torch.tensor([self.tokenizer.mask_id] * 2),
            "labels": torch.tensor(native_ids),
            "target_len": 2,
        }]

        payload = run_generative_eval(
            FakeModel(),
            self.tokenizer,
            dataset,
            device=torch.device("cpu"),
            decoding="iterative",
            num_samples=2,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            num_examples=-1,
            seed=42,
            skip_perplexity=True,
        )

        self.assertEqual(payload["records"][0]["target_len"], 2)
        self.assertEqual(
            payload["records"][0]["length_policy"]["source"],
            "native_mask_count",
        )
        self.assertEqual(payload["length_policy"]["mode"], "fixed")

        self.assertEqual(payload["records"][0]["samples"], ["WY", "WW"])
        self.assertEqual(shared_decode.call_args.kwargs["decoding"], "iterative")
        self.assertEqual(len(shared_decode.call_args.args[5]), 2)


    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_held_out_eval_uses_the_requested_anchor_trim(self, shared_decode):
        native_ids = self.tokenizer.encode("ACDEFG")
        shared_decode.return_value = [self.tokenizer.encode("AYDEFG")]
        dataset = [{
            "input_ids": torch.tensor([self.tokenizer.mask_id] * 6),
            "labels": torch.tensor(native_ids),
            "target_len": 6,
        }]

        payload = run_generative_eval(
            FakeModel(),
            self.tokenizer,
            dataset,
            device=torch.device("cpu"),
            decoding="independent",
            num_samples=1,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            num_examples=-1,
            seed=42,
            skip_perplexity=True,
            anchor_trim=1,
        )

        # With one residue trimmed at each end, positions 1..4 are core and only position 1
        # differs: 3/4. The old hard-coded trim of two would instead report 2/2.
        self.assertEqual(payload["anchor_trim_each_end"], 1)
        self.assertAlmostEqual(
            payload["records"][0]["metrics"]["core_recovery_mean"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
