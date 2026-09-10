"""Trainer lifecycle and finished-run artifacts shared by every method."""
import logging
from pathlib import Path

from transformers import TrainerCallback

from cdrseqgen.train.record_results import plot_training_curves, save_training_csv

logger = logging.getLogger(__name__)


class SaveTokenizerCallback(TrainerCallback):
    """Each intermediate checkpoint carries the exact vocabulary used to train it."""
    def __init__(self, save_tokenizer):
        self.save_tokenizer = save_tokenizer

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            self.save_tokenizer(str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))
        return control


def run_training_pipeline(trainer, training_args, *, tokenizer=None):
    """Train/evaluate and persist the same metric artifacts for every method.

    A method owns its model, collator, and metrics; it should not own whether ``train_results.json``
    or ``eval_results.json`` gets written. Keeping that lifecycle here prevents a new method from
    looking complete in ``epoch_metrics.csv`` while silently omitting the standard HF artifacts.
    """
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if training_args.do_train:
        if callable(save_tokenizer):
            trainer.add_callback(SaveTokenizerCallback(save_tokenizer))
        logger.info("🚀 Starting Training...")
        train_result = trainer.train()
        trainer.save_model(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        # Generic causal tokenizers may gain the infilling sentinels at load time. Save that exact
        # mapping beside the resized model so the checkpoint is independently reloadable.
        if callable(save_tokenizer):
            save_tokenizer(training_args.output_dir)

    if training_args.do_eval:
        logger.info("🔍 Starting Evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    save_training_csv(trainer, training_args.output_dir)
    plot_training_curves(training_args.output_dir)
