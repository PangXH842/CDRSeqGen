import unittest
from unittest.mock import patch

import torch

from benchmark.adapters.iglm import IgLMSampler
from cdrseqgen.data.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.evaluation import run_generative_eval
from cdrseqgen.eval.metrics import calculate
from cdrseqgen.methods.mlm.generation import GeneratedSpan


class FrameworkDataset:
    def __init__(self):
        self.parts = {
            "pt1": "AC",
            "target": "AAAA",
            "target_length": 4,
            "pt2": "WY",
            "partner_chain": None,
            "antigen": None,
            "max_len": 32,
        }

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {"target_len": 4}

    def framework_parts(self, _index):
        return dict(self.parts)


class EvalModel:
    def eval(self):
        return self


class VariableLengthEvalTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_unequal_lengths_use_edit_metrics_and_length_matched_aar(self):
        metrics = calculate(["AAAA", "AAA"], native="AAAA")
        self.assertEqual(metrics["length_match_rate"], 0.5)
        self.assertEqual(metrics["aar_mean"], 1.0)
        self.assertEqual(metrics["aar_length_matched_mean"], 1.0)
        self.assertAlmostEqual(metrics["length_absolute_error_mean"], 0.5)
        self.assertGreater(metrics["edit_similarity_mean"], 0.8)

    def test_iglm_natural_prepare_does_not_consult_native_length_limit(self):
        sampler = IgLMSampler.__new__(IgLMSampler)
        sampler.length_policy = "natural"
        sampler.max_length = 5
        sampler._starting_tokens = lambda _record: torch.zeros((1, 4), dtype=torch.long)
        sampler.prepare({"example_id": "x", "native_length": 100})
        self.assertIsNone(sampler.target_total_length)

    @patch("cdrseqgen.methods.mlm.variable_length_evaluation.sample_length_conditioned_spans")
    def test_held_out_inferred_mode_keeps_truth_scoring_only(self, generate):
        ids4 = self.tokenizer.encode("AAAA")
        ids3 = self.tokenizer.encode("AAA")
        generate.return_value = (
            [
                GeneratedSpan(ids4, 4, [4] * 4, list(range(4)), {}),
                GeneratedSpan(ids3, 3, [4] * 3, list(range(3)), {}),
            ],
            {"mode": "inferred", "source": "length_predictor", "selected_lengths": [4, 3]},
        )

        payload = run_generative_eval(
            EvalModel(),
            self.tokenizer,
            FrameworkDataset(),
            device=torch.device("cpu"),
            length_mode="inferred",
            length_predictor=object(),
            num_samples=2,
            num_examples=-1,
            skip_perplexity=True,
        )

        record = payload["records"][0]
        self.assertEqual(record["generated_lengths"], [4, 3])
        self.assertEqual(record["metrics"]["length_match_rate"], 0.5)
        self.assertEqual(record["metrics"]["aar_mean"], 1.0)
        call = generate.call_args.kwargs
        self.assertEqual(call["policy"].mode, "inferred")
        self.assertNotIn("native", call)


if __name__ == "__main__":
    unittest.main()
