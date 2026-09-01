import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.data.dataset_mlm import AntibodyTokenizer, TokenizerAdapter
from cdrseqgen.methods.mlm.length_predictor import fit_length_predictor


class TinyFrameworkDataset:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def framework_parts(self, index):
        pt1, length = self.rows[index]
        return {
            "pt1": pt1,
            "target": "A" * length,
            "target_length": length,
            "pt2": "WY",
            "partner_chain": None,
            "antigen": None,
            "max_len": 16,
        }


class TinyHiddenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(25, 4)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False,
                return_dict=True):
        hidden = self.embedding(input_ids)
        return {"logits": hidden, "hidden_states": (hidden,)}


class LengthPredictorFitTests(unittest.TestCase):
    def test_frozen_backbone_fit_writes_loadable_artifacts_with_empty_eval(self):
        tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        model = TinyHiddenModel()
        before = model.embedding.weight.detach().clone()
        train = TinyFrameworkDataset([("AC", 2), ("AC", 2), ("DE", 3), ("DE", 3)])
        empty_eval = TinyFrameworkDataset([])

        with tempfile.TemporaryDirectory() as directory:
            predictor = fit_length_predictor(
                model,
                tokenizer,
                train,
                empty_eval,
                output_dir=directory,
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.01,
                batch_size=2,
            )
            path = Path(directory)
            self.assertTrue((path / LENGTH_CONFIG_FILENAME).exists())
            self.assertTrue((path / LENGTH_HEAD_FILENAME).exists())
            config = json.loads((path / LENGTH_CONFIG_FILENAME).read_text())

        self.assertEqual(predictor.lengths, (2, 3))
        self.assertTrue(torch.equal(before, model.embedding.weight.detach()))
        self.assertEqual(config["train_length_counts"], {"2": 2, "3": 2})
        self.assertEqual(config["train_mode_length"], 2)
        self.assertIsNone(config["eval_accuracy"])
        self.assertIsNone(config["eval_mode_baseline_accuracy"])
        self.assertIsNone(config["eval_length_mae"])

    def test_eval_diagnostics_include_mode_baseline_and_mae(self):
        tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        model = TinyHiddenModel()
        train = TinyFrameworkDataset([("AC", 2), ("DE", 2), ("FG", 3)])
        evaluation = TinyFrameworkDataset([("AC", 2), ("DE", 3), ("FG", 4)])

        with tempfile.TemporaryDirectory() as directory:
            predictor = fit_length_predictor(
                model,
                tokenizer,
                train,
                evaluation,
                output_dir=directory,
                device=torch.device("cpu"),
                epochs=1,
                learning_rate=0.01,
                batch_size=2,
            )
            config = predictor.config

        self.assertEqual(config["eval_examples"], 3)
        self.assertEqual(config["eval_supported_examples"], 2)
        self.assertEqual(config["eval_unseen_lengths"], [4])
        self.assertAlmostEqual(config["eval_mode_baseline_accuracy"], 1 / 3)
        self.assertAlmostEqual(config["eval_mode_baseline_length_mae"], 1.0)
        self.assertIsInstance(config["eval_accuracy"], float)
        self.assertIsInstance(config["eval_supported_accuracy"], float)
        self.assertIsInstance(config["eval_length_mae"], float)


if __name__ == "__main__":
    unittest.main()
