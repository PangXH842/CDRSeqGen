"""Root-to-method command routing with model loading and execution mocked."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import evaluate
import generate
import train


def method_with_modules(**modules):
    return SimpleNamespace(name="mlm", module=Mock(side_effect=modules.__getitem__))


class RootCommandDispatchTests(unittest.TestCase):
    def test_training_dispatch_preserves_arguments_for_each_method(self):
        for name in train.METHODS:
            with self.subTest(method=name):
                argv = ["train.py", "--method", name, "--seed", "43"]
                observed = []
                command = SimpleNamespace(main=Mock(side_effect=lambda: observed.append(list(train.sys.argv))))
                method = method_with_modules(train_cli=command)
                with patch("sys.argv", argv), patch("train.resolve", return_value=method) as resolve:
                    train.main()
                resolve.assert_called_once_with(name)
                method.module.assert_called_once_with("train_cli")
                command.main.assert_called_once_with()
                self.assertEqual(observed, [argv])

    def test_generation_forwards_case_order_and_method_state(self):
        cases = [{"name": "first"}, {"name": "second"}]
        config = {"model_type": "scratch", "max_len": 64}
        args = SimpleNamespace(model_path="/tmp/checkpoint", max_len=None,
                               seed=43, force=False, output="none")
        command = Mock()
        command.prepare_generation.return_value = sentinel.policies
        command.generation_state.return_value = sentinel.state
        model, tokenizer = Mock(), Mock()
        backbone = Mock()
        backbone.build_tokenizer.return_value = tokenizer
        backbone.load_pretrained_model.return_value = model
        backbones = SimpleNamespace(resolve=Mock(return_value=backbone))
        method = method_with_modules(gen_cli=command, backbones=backbones)

        with patch("generate.get_configs", return_value=config), \
                patch("generate.get_test_case", return_value=cases), \
                patch("generate.check_compatibility", return_value=([], [])), \
                patch("generate.checkpoint_method", return_value=method), \
                patch("generate.torch.cuda.is_available", return_value=False), \
                patch("builtins.print"):
            generate.main(args)

        self.assertEqual(args.max_len, 64)
        command.prepare_generation.assert_called_once_with(args, cases, config)
        self.assertIs(command.generation_state.call_args.args[1], sentinel.policies)
        self.assertEqual(len(command.generate_case.call_args_list), 2)
        for invocation, case in zip(command.generate_case.call_args_list, cases):
            self.assertIs(invocation.args[0], model)
            self.assertIs(invocation.args[1], tokenizer)
            self.assertIs(invocation.args[3], args)
            self.assertIs(invocation.args[5], case)
            self.assertIs(invocation.args[6], sentinel.state)
            self.assertIs(invocation.args[7], generate._print_case)

    def test_invalid_generation_options_stop_before_weights_are_loaded(self):
        command = Mock()
        command.prepare_generation.side_effect = ValueError("unsupported length policy")
        backbones = Mock()
        method = method_with_modules(gen_cli=command, backbones=backbones)
        args = SimpleNamespace(model_path="/tmp/checkpoint", max_len=64, force=False)
        with patch("generate.get_configs", return_value={"model_type": "scratch"}), \
                patch("generate.get_test_case", return_value=[{}]), \
                patch("generate.check_compatibility", return_value=([], [])), \
                patch("generate.checkpoint_method", return_value=method), \
                patch("builtins.print"):
            with self.assertRaisesRegex(SystemExit, "unsupported length policy"):
                generate.main(args)
        method.module.assert_called_once_with("gen_cli")
        backbones.resolve.assert_not_called()

    def test_evaluation_forwards_method_options_and_keeps_contexts_separate(self):
        command = Mock()
        command.evaluation_options.return_value = {
            "length_mode": "inferred", "length_predictor": sentinel.length_head,
        }
        model, tokenizer = Mock(), Mock()
        dataset = [sentinel.example]
        backbone = Mock()
        backbone.load_pretrained_model.return_value = model
        backbones = SimpleNamespace(resolve=Mock(return_value=backbone))
        training = SimpleNamespace(get_datasets=Mock(return_value=([], dataset, tokenizer)))
        arguments = SimpleNamespace(ModelArguments=object, DataArguments=object,
                                    GenerationArguments=object)
        method = method_with_modules(eval_cli=command, backbones=backbones,
                                     training=training, arguments=arguments)
        model_args = SimpleNamespace(model_type="scratch", anchor_trim=3)
        data_args = SimpleNamespace(data_path="/tmp/corpus.json", target_region="h3",
                                    eval_split="validation", include_partner_chain=True,
                                    include_antigen=False, context_dropout=0.5)
        parsed = (model_args, data_args, SimpleNamespace(), SimpleNamespace())
        run_config = {"method": "mlm", "target_region": "h3"}

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"method": "mlm"}))
            argv = ["evaluate.py", "--config", str(config), "--model_dir", directory,
                    "--output", "none", "--judge", "", "--partner_context", "without",
                    "--judge_partner_context", "with", "--skip_perplexity"]
            with patch("sys.argv", argv), \
                    patch("evaluate.resolve_method", return_value=method), \
                    patch("evaluate.checkpoint_method"), \
                    patch("evaluate.read_run_config", return_value=run_config), \
                    patch("evaluate.HfArgumentParser"), \
                    patch("evaluate.parse_config_dict", return_value=parsed), \
                    patch("evaluate.compatibility_problems", return_value=[]), \
                    patch("evaluate.evaluation_provenance", return_value={}), \
                    patch("evaluate.evaluation_context_view", return_value=dataset) as context_view, \
                    patch("evaluate.torch.cuda.is_available", return_value=False), \
                    patch("builtins.print"):
                evaluate.main()

        context_view.assert_called_once_with(dataset, include_partner_chain=False)
        command.evaluate.assert_called_once()
        invocation = command.evaluate.call_args
        self.assertEqual(invocation.args, (model, tokenizer, dataset))
        self.assertEqual(invocation.kwargs["length_mode"], "inferred")
        self.assertIs(invocation.kwargs["length_predictor"], sentinel.length_head)
        self.assertEqual(invocation.kwargs["anchor_trim"], 3)
        self.assertFalse(invocation.kwargs["meta"]["include_partner_chain"])
        self.assertTrue(invocation.kwargs["meta"]["judge_include_partner_chain"])
        self.assertEqual(invocation.kwargs["meta"]["eval_context_dropout"], 0.0)


if __name__ == "__main__":
    unittest.main()
