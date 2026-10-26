import unittest
from unittest.mock import patch

from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.length import prepare_fixed_length_prompt
from cdrseqgen.methods.mlm.sampling import sample_masked_span


class FixedLengthMLMGenerationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())

    def test_prompt_preserves_exact_mask_count_and_framework_order(self):
        prompt = prepare_fixed_length_prompt(
            self.tokenizer, "AC", "WY", 3, max_len=20
        )

        expected = (
            self.tokenizer.encode("AC")
            + [self.tokenizer.mask_id] * 3
            + self.tokenizer.encode("WY")
        )
        self.assertEqual(prompt.input_ids, expected)
        self.assertEqual(prompt.mask_positions, [2, 3, 4])
        self.assertEqual(prompt.target_length, 3)
        self.assertEqual(prompt.metadata()["input_tokens"], 7)

    def test_prompt_reports_training_compatible_context_truncation(self):
        prompt = prepare_fixed_length_prompt(
            self.tokenizer, "ACDE", "FGHI", 3, max_len=7
        )

        self.assertEqual(len(prompt.input_ids), 7)
        self.assertEqual(prompt.pt1_tokens, 4)
        self.assertEqual(prompt.pt2_tokens, 0)
        self.assertEqual(prompt.pt1_truncated_tokens, 0)
        self.assertEqual(prompt.pt2_truncated_tokens, 4)

    @patch("cdrseqgen.methods.mlm.sampling.sample_iterative")
    @patch("cdrseqgen.methods.mlm.sampling.sample_independent")
    def test_decoder_dispatch_preserves_sampling_arguments(self, independent, iterative):
        independent.return_value = [[1, 2]]
        result = sample_masked_span(
            model="model",
            tokenizer=self.tokenizer,
            device="cpu",
            generator="generator",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            decoding="independent",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
        )

        self.assertEqual(result, [[1, 2]])
        independent.assert_called_once_with(
            model="model",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            banned_ids=self.tokenizer.special_ids,
            device="cpu",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
            generator="generator",
        )
        iterative.assert_not_called()

        iterative.return_value = [[2, 1]]
        result = sample_masked_span(
            model="model",
            tokenizer=self.tokenizer,
            device="cpu",
            generator="generator",
            input_ids=[4, 4],
            mask_positions=[0, 1],
            decoding="iterative",
            num_samples=5,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
        )

        self.assertEqual(result, [[2, 1]])
        iterative.assert_called_once()


if __name__ == "__main__":
    unittest.main()
