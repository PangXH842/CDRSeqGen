import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.length import sample_length_conditioned_spans
from cdrseqgen.methods.mlm.length import resolve_length_policy
from cdrseqgen.methods.mlm.length import (
    SCHEMA_VERSION,
    LengthHead,
    LengthPredictor,
)


class HiddenModel:
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False,
                 return_dict=True):
        batch, length = input_ids.shape
        hidden = torch.ones(batch, length, 2, device=input_ids.device)
        return {"logits": torch.zeros(batch, length, 25), "hidden_states": (hidden,)}


class FixedSelectionPredictor:
    def select(self, *_args, **_kwargs):
        return [2, 3, 2], {2: 0.6, 3: 0.4}, type("Prompt", (), {
            "input_ids": [4], "gap_position": 0, "reserve_length": 3,
        })()


class LengthPredictorTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.config = {
            "schema_version": SCHEMA_VERSION,
            "feature_dim": 2,
            "supported_lengths": [2, 3, 4],
            "reserve_length": 4,
        }
        head = LengthHead(2, 3)
        with torch.no_grad():
            head.normalization.weight.fill_(1)
            head.normalization.bias.zero_()
            head.classifier.weight.zero_()
            head.classifier.bias.copy_(torch.tensor([0.0, 2.0, 5.0]))
        self.predictor = LengthPredictor(head, self.config)

    def test_bounds_mask_classes_before_selection(self):
        policy = resolve_length_policy(min_length=2, max_length=3)
        selected, probabilities, _prompt = self.predictor.select(
            HiddenModel(), self.tokenizer, torch.device("cpu"),
            torch.Generator().manual_seed(1),
            pt1="AC", pt2="WY", policy=policy, num_samples=3, temperature=0,
        )
        self.assertEqual(selected, [3, 3, 3])
        self.assertNotIn(4, probabilities)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=6)

    def test_impossible_bounds_fail_with_observed_support(self):
        policy = resolve_length_policy(min_length=7)
        with self.assertRaisesRegex(ValueError, "contain no class"):
            self.predictor.select(
                HiddenModel(), self.tokenizer, torch.device("cpu"),
                torch.Generator().manual_seed(1),
                pt1="AC", pt2="WY", policy=policy, num_samples=1,
            )

    def test_artifact_roundtrip(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_file(self.predictor.head.state_dict(), str(path / LENGTH_HEAD_FILENAME))
            (path / LENGTH_CONFIG_FILENAME).write_text(json.dumps(self.config))
            loaded = LengthPredictor.load(path, device=torch.device("cpu"))
            self.assertEqual(loaded.lengths, (2, 3, 4))

    @patch("cdrseqgen.methods.mlm.length.sample_masked_span")
    def test_variable_generator_groups_draws_by_selected_length(self, sample_span):
        def sample_for_prompt(*args, **kwargs):
            length = len(args[5])
            count = kwargs["num_samples"]
            return [self.tokenizer.encode("A" * length) for _ in range(count)]

        sample_span.side_effect = sample_for_prompt
        generated, metadata = sample_length_conditioned_spans(
            model=HiddenModel(), tokenizer=self.tokenizer, device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(2),
            pt1="AC", pt2="WY", policy=resolve_length_policy(),
            length_predictor=FixedSelectionPredictor(), decoding="independent",
            num_samples=3, temperature=1.0, top_k=0, top_p=1.0,
        )
        self.assertEqual([span.length for span in generated], [2, 3, 2])
        self.assertEqual(metadata["selected_length_counts"], {"2": 2, "3": 1})
        self.assertEqual(sample_span.call_count, 2)


if __name__ == "__main__":
    unittest.main()
