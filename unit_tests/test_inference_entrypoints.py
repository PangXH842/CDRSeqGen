"""Exercise root -> MLM command -> workflow -> real synthetic candidate scoring."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import evaluate
import generate
from cdrseqgen.datasets.dataset_mlm import (
    AntibodyTokenizer, MaskedInfillingDataset, TokenizerAdapter,
)


class InferenceEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerAdapter.from_custom(AntibodyTokenizer())
        self.model = Mock()
        residue = self.tokenizer.encode("A")[0]

        def forward(*, input_ids):
            logits = torch.zeros(*input_ids.shape, 32)
            logits[..., residue] = 10
            return {"logits": logits}

        self.model.side_effect = forward
        self.backbone = Mock()
        self.backbone.build_tokenizer.return_value = self.tokenizer
        self.backbone.load_pretrained_model.return_value = self.model

    def test_generation_runs_fixed_case_and_writes_compatibility_without_length_head(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"method": "mlm", "model_type": "scratch", "target_region": "h3",
                      "max_len": 32, "include_partner_chain": False, "include_antigen": False,
                      "source": "cli"}
            (root / "run_config.json").write_text(json.dumps(config))
            output = root / "generated.json"
            args = SimpleNamespace(
                model_dir=directory, model_type=None, max_len=None, target_loop=None,
                test_file=None, pt1="AC", pt2="DE", target_truth="WY",
                target_length=None, min_length=2, max_length=2, use_native_length=False,
                partner_chain=None, antigen=None, max_new_tokens=None,
                decoding=None, num_samples=2, temperature=0.0,
                top_k=0, top_p=1.0, seed=42, skip_perplexity=True,
                show_samples=0, force=False, output=str(output),
            )
            with patch("cdrseqgen.methods.mlm.backbones.resolve", return_value=self.backbone), \
                    patch("cdrseqgen.methods.mlm.length.LengthPredictor.load") as load_head, \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                results = generate.main(args)
            payload = json.loads(output.read_text())

        load_head.assert_not_called()
        self.assertEqual(args.max_len, 32)
        self.assertEqual(args.decoding, "independent")
        self.assertEqual(self.model.call_count, 1)
        self.assertEqual(payload["cases"], results)
        self.assertEqual(payload["cases"][0]["samples"], [
            {"seq": "AA", "length": 2, "length_match": True, "recovery": 0.0},
        ])
        self.assertEqual(payload["cases"][0]["length_policy"]["source"], "explicit_bounds")
        self.assertEqual(payload["model"]["compatibility"], {
            "problems": [], "warnings": [], "forced": False,
        })
        self.assertEqual(payload["model"]["expects"]["source"], "run_config")

    def test_evaluation_scores_fixed_native_and_refuses_repeated_output_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chain = dict(fwr1="AC", cdr1="D", fwr2="E", cdr2="F", fwr3="G",
                         cdr3="AW", fwr4="WY", full_aa="ACDEFGAWWY")
            data_path = root / "synthetic.json"
            data_path.write_text(json.dumps([
                {"heavy": chain, "light": chain, "split": "validation"},
            ]))
            dataset = MaskedInfillingDataset(str(data_path), self.tokenizer, max_len=32)
            config = {
                "method": "mlm", "model_type": "scratch", "target_region": "h3",
                "data_path": str(data_path), "eval_split": "validation",
                "max_len": 32, "masking_policy": "full_span",
                "include_partner_chain": False, "include_antigen": False,
                "output_dir": directory, "report_to": "none", "use_cpu": True,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            (root / "run_config.json").write_text(json.dumps(config))
            output = root / "evaluated.json"
            argv = ["evaluate.py", "--config", str(config_path), "--model_dir", directory,
                    "--output", str(output), "--num_examples", "-1", "--num_samples", "2",
                    "--temperature", "0", "--skip_perplexity"]
            with patch("sys.argv", argv), \
                    patch("cdrseqgen.methods.mlm.backbones.resolve", return_value=self.backbone), \
                    patch("cdrseqgen.methods.mlm.training.get_datasets", return_value=([], dataset, self.tokenizer)) as get_data, \
                    patch("cdrseqgen.methods.mlm.length.LengthPredictor.load") as load_head, \
                    patch("torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                payload = evaluate.main()
                with self.assertRaisesRegex(SystemExit, "Refusing to overwrite"):
                    evaluate.main()
            saved = json.loads(output.read_text())

        load_head.assert_not_called()
        get_data.assert_called_once()
        self.backbone.load_pretrained_model.assert_called_once()
        self.assertEqual(payload["records"], saved["records"])
        self.assertEqual(saved["records"][0]["samples"], ["AA"])
        self.assertEqual(saved["records"][0]["metrics"]["aar_mean"], 0.5)
        self.assertEqual(saved["length_policy"]["source"], "native_mask_count")
        self.assertEqual(saved["compatibility"], {"problems": [], "forced": False})

    def test_ar_s2s_generation_entries_reject_fixed_bounds_before_loading(self):
        for method in ["ar", "s2s"]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = {"method": method, "model_type": "scratch", "target_region": "h3",
                          "max_len": 32, "include_partner_chain": False, "include_antigen": False}
                (Path(directory) / "run_config.json").write_text(json.dumps(config))
                args = SimpleNamespace(
                    model_dir=directory, model_type=None, max_len=None, target_loop=None,
                    test_file=None, pt1="AC", pt2="DE", target_truth=None,
                    target_length=None, min_length=2, max_length=2, use_native_length=False,
                    partner_chain=None, antigen=None, max_new_tokens=None,
                    decoding=None, force=False,
                )
                with patch(f"cdrseqgen.methods.{method}.backbones.resolve") as backbone, \
                        patch("builtins.print"):
                    with self.assertRaisesRegex(SystemExit, "EOS"):
                        generate.main(args)
                backbone.assert_not_called()

    def test_ar_s2s_evaluation_entries_reject_native_length_before_loading(self):
        for method in ["ar", "s2s"]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = {"method": method}
                path = root / "config.json"
                path.write_text(json.dumps(config))
                (root / "run_config.json").write_text(json.dumps(config))
                argv = ["evaluate.py", "--config", str(path), "--model_dir", directory,
                        "--length_mode", "fixed_native", "--output", "none"]
                with patch("sys.argv", argv), \
                        patch(f"cdrseqgen.methods.{method}.backbones.resolve") as backbone:
                    with self.assertRaisesRegex(ValueError, "EOS"):
                        evaluate.main()
                backbone.assert_not_called()


if __name__ == "__main__":
    unittest.main()
