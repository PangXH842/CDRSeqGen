"""MLM prompt budgets, mixed-precision heads and nested judge views."""
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import Subset

from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.eval.generative import build_masked_judge_prompt
from cdrseqgen.methods.mlm.length import (
    FixedNativeStrategy, VariableLengthStrategy, LengthHead, LengthPredictor,
    prepare_fixed_length_prompt, prepare_gap_prompt, resolve_length_policy,
    sample_length_conditioned_spans,
)


class Frameworks:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return dict(input_ids=torch.tensor([index + 5, 4, 6]),
                    labels=torch.tensor([-100, 5, -100]), target_len=1)

    def framework_parts(self, index):
        return dict(pt1=("AC", "DE", "FG", "HI", "KL", "MN")[index], pt2="PQ",
                    target="A", target_length=1, max_len=64,
                    partner_chain=None, antigen=None)


class BfloatModel:
    config = SimpleNamespace(max_position_embeddings=16)

    def __call__(self, input_ids, **kwargs):
        hidden = torch.ones(*input_ids.shape, 2, dtype=torch.bfloat16)
        return {"hidden_states": (hidden,)}


class LengthContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_fixed_and_reserved_spans_cannot_exceed_the_input_budget(self):
        for build in (lambda: prepare_fixed_length_prompt(self.tokenizer, "AC", "DE", 20, max_len=10),
                      lambda: prepare_gap_prompt(self.tokenizer, "AC", "DE", reserve_length=20, max_len=10)):
            with self.subTest(build=build):
                with self.assertRaisesRegex(ValueError, "requires.*allows"):
                    build()
        self.tokenizer.uses_special_tokens = True
        with self.assertRaisesRegex(ValueError, "requires.*allows"):
            prepare_fixed_length_prompt(self.tokenizer, "", "", 9, max_len=10)

    def test_declared_tokenizer_and_model_capacities_are_enforced_before_forward(self):
        self.tokenizer.max_positions = 8
        with self.assertRaisesRegex(ValueError, "max_positions"):
            prepare_fixed_length_prompt(self.tokenizer, "", "", 9, max_len=32)
        del self.tokenizer.max_positions
        model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=3))
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            sample_length_conditioned_spans(
                model, self.tokenizer, "cpu", torch.Generator(), pt1="AC", pt2="DE",
                policy=resolve_length_policy(target_length=2), max_len=16,
                decoding="independent", num_samples=1, temperature=0, top_k=0, top_p=1)

    def test_bfloat_features_work_for_head_training_and_length_inference(self):
        head = LengthHead(2, 3)
        features = torch.tensor([[0.1, 0.5]], dtype=torch.bfloat16, requires_grad=True)
        actual = head(features)
        torch.testing.assert_close(actual, head(features.float()), rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(head.classifier.weight.grad).all())
        predictor = LengthPredictor(head, dict(supported_lengths=[1, 2, 3], reserve_length=3))
        selected, probabilities, _ = predictor.select(
            BfloatModel(), self.tokenizer, "cpu", torch.Generator().manual_seed(5),
            pt1="AC", pt2="DE", policy=resolve_length_policy(), max_len=16,
            num_samples=2, temperature=0)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in probabilities.values()))

    def test_length_fitting_checks_the_reserved_span_against_model_capacity(self):
        from unittest.mock import Mock
        from cdrseqgen.methods.mlm.length import _extract_dataset_features
        model = Mock(config=SimpleNamespace(max_position_embeddings=16))
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            _extract_dataset_features(model, self.tokenizer, Frameworks(), device="cpu",
                                      batch_size=2, reserve_length=20)
        model.assert_not_called()

    def test_both_strategies_map_nested_subsets_to_the_correct_judge_framework(self):
        full = Frameworks()
        nested = Subset(Subset(full, [4, 1, 5]), [2, 0])
        for judge in (full, nested):
            options = dict(model=object(), tokenizer=self.tokenizer, device="cpu",
                           generator=torch.Generator(), eval_dataset=nested, judge_dataset=judge,
                           judge_tokenizer=self.tokenizer, decoding="independent", num_samples=1,
                           temperature=0, top_k=0, top_p=1)
            fixed = FixedNativeStrategy(**options)
            ids, _ = fixed.judge_inputs(0, "A")
            self.assertEqual(ids[0], 10)
            variable = VariableLengthStrategy(length_predictor=object(),
                length_policy=resolve_length_policy(), length_temperature=1, **options)
            self.assertEqual(variable.judge_inputs(0, "AC"),
                             build_masked_judge_prompt(self.tokenizer, full.framework_parts(5), 2))


if __name__ == "__main__":
    unittest.main()
