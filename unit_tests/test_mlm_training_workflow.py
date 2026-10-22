"""MLM command/workflow contracts; model, training and generation are mocked."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from cdrseqgen.methods.mlm import train_cli, training


class MLMCommandTests(unittest.TestCase):
    def test_command_configures_logging_parses_and_delegates(self):
        arguments = (sentinel.model_args, sentinel.data_args,
                     sentinel.gen_args, sentinel.training_args)
        events = Mock()
        with patch.object(train_cli, "configure_logging") as logging, \
                patch.object(train_cli, "setup_args", return_value=arguments) as parse, \
                patch.object(train_cli, "run_training") as run:
            events.attach_mock(logging, "logging")
            events.attach_mock(parse, "parse")
            events.attach_mock(run, "run")
            self.assertIsNone(train_cli.main())
            run.assert_called_once_with(*arguments)
        self.assertEqual([call[0] for call in events.mock_calls], ["logging", "parse", "run"])


class MLMTrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.model_args = SimpleNamespace(
            model_type="scratch", hidden_size=128, num_layers=4,
            num_attention_heads=8, intermediate_size=512, loss_type="weighted",
            core_loss_weight=3., anchor_loss_weight=1., anchor_trim=2,
        )
        self.data_args = SimpleNamespace(
            max_len=384, train_length_predictor=True, length_predictor_epochs=20,
            length_predictor_learning_rate=0.01, length_predictor_batch_size=16,
        )
        self.gen_args = SimpleNamespace(run_generative_eval=True)
        self.training_args = SimpleNamespace(
            seed=42, output_dir="/tmp/mock-mlm-workflow", device="cpu",
            do_train=True, do_eval=True, label_smoothing_factor=0.1,
        )
        self.arguments = (self.model_args, self.data_args, self.gen_args, self.training_args)
        self.tokenizer = SimpleNamespace(vocab_size=25, pad_id=7)
        self.train_dataset, self.eval_dataset = [sentinel.train], [sentinel.eval]
        self.trainer = SimpleNamespace(model=sentinel.initial_model)
        self.backbone = SimpleNamespace(load_training_model=Mock(return_value=sentinel.initial_model))
        self.events = Mock()

        self.seed = self.mock_stage("set_seed")
        self.resolve = self.mock_stage("resolve", return_value=self.backbone)
        self.datasets = self.mock_stage(
            "get_datasets", return_value=(self.train_dataset, self.eval_dataset, self.tokenizer))
        self.latest = self.mock_stage("update_latest_link")
        self.events.attach_mock(self.backbone.load_training_model, "load_model")
        self.callbacks = self.mock_stage("build_callbacks", return_value=[sentinel.callback])
        self.make_trainer = self.mock_stage("MLMLossTrainer", return_value=self.trainer)
        self.lifecycle = self.mock_stage("run_training_pipeline", side_effect=self.finish_training)
        self.fit = self.enterContext(patch(
            "cdrseqgen.methods.mlm.length_predictor.fit_length_predictor",
            return_value=sentinel.length_predictor))
        self.events.attach_mock(self.fit, "fit_length_predictor")
        self.generate = self.mock_stage("post_training_generation")

    def mock_stage(self, name, **kwargs):
        mocked = self.enterContext(patch.object(training, name, **kwargs))
        self.events.attach_mock(mocked, name)
        return mocked

    def finish_training(self, trainer, args, *, tokenizer):
        if args.do_train:
            trainer.model = sentinel.selected_model

    def test_full_workflow_preserves_order_and_uses_selected_model_for_length_head(self):
        self.assertIs(training.run_training(*self.arguments), self.trainer)
        self.assertEqual([call[0] for call in self.events.mock_calls], [
            "set_seed", "resolve", "get_datasets", "update_latest_link", "load_model",
            "build_callbacks", "MLMLossTrainer", "run_training_pipeline",
            "fit_length_predictor", "post_training_generation",
        ])
        self.seed.assert_called_once_with(42)
        self.resolve.assert_called_once_with("scratch")
        self.datasets.assert_called_once_with(
            self.model_args, self.data_args, self.training_args, self.backbone)
        self.latest.assert_called_once_with(self.training_args.output_dir)
        self.backbone.load_training_model.assert_called_once_with(
            "scratch", vocab_size=25, hidden_size=128, num_layers=4,
            num_attention_heads=8, intermediate_size=512, max_len=384)

        options = self.make_trainer.call_args.kwargs
        self.assertIs(options["model"], sentinel.initial_model)
        self.assertIs(options["args"], self.training_args)
        self.assertIs(options["train_dataset"], self.train_dataset)
        self.assertIs(options["eval_dataset"], self.eval_dataset)
        self.assertEqual(options["loss_config"].loss_type, "weighted_cross_entropy")
        self.assertEqual(options["loss_config"].label_smoothing, 0.1)
        self.assertIs(options["data_collator"].func, training.mlm_collate_fn)
        self.assertEqual(options["data_collator"].keywords, {"pad_token_id": 7})
        self.assertIs(options["compute_metrics"].func, training.compute_metrics)
        self.assertEqual(options["compute_metrics"].keywords, {"trim": 2})
        self.assertEqual(options["callbacks"], [sentinel.callback])
        self.lifecycle.assert_called_once_with(
            self.trainer, self.training_args, tokenizer=self.tokenizer)
        self.fit.assert_called_once_with(
            sentinel.selected_model, self.tokenizer, self.train_dataset, self.eval_dataset,
            output_dir=self.training_args.output_dir, device="cpu", epochs=20,
            learning_rate=0.01, batch_size=16, seed=42)
        self.generate.assert_called_once_with(
            self.trainer, self.tokenizer, self.eval_dataset,
            self.data_args, self.gen_args, self.training_args,
            anchor_trim=2, length_predictor=sentinel.length_predictor)

    def test_evaluation_only_skips_length_fitting_and_keeps_generation_available(self):
        self.training_args.do_train = False
        training.run_training(*self.arguments)
        self.lifecycle.assert_called_once()
        self.fit.assert_not_called()
        self.assertIs(self.trainer.model, sentinel.initial_model)
        self.assertIsNone(self.generate.call_args.kwargs["length_predictor"])

    def test_extensions_can_both_be_disabled(self):
        self.data_args.train_length_predictor = False
        self.gen_args.run_generative_eval = False
        training.run_training(*self.arguments)
        self.lifecycle.assert_called_once()
        self.fit.assert_not_called()
        self.generate.assert_not_called()

    def test_length_fitting_does_not_depend_on_generation_being_enabled(self):
        self.gen_args.run_generative_eval = False
        training.run_training(*self.arguments)
        self.fit.assert_called_once()
        self.generate.assert_not_called()

    def test_dataset_failure_leaves_latest_and_model_untouched(self):
        self.datasets.side_effect = ValueError("invalid evaluation split")
        with self.assertRaisesRegex(ValueError, "invalid evaluation split"):
            training.run_training(*self.arguments)
        self.latest.assert_not_called()
        self.backbone.load_training_model.assert_not_called()
        self.make_trainer.assert_not_called()
        self.lifecycle.assert_not_called()
        self.fit.assert_not_called()
        self.generate.assert_not_called()

    def test_training_failure_stops_before_post_training_extensions(self):
        self.lifecycle.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            training.run_training(*self.arguments)
        self.fit.assert_not_called()
        self.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
