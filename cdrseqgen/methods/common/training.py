"""Trainer workflow and finished-run artifacts shared by methods.

Classes:
  * ExternalLossTrainer -- forward model inputs, then call a method's loss hook
  * SaveTokenizerCallback -- save the tokenizer with each training checkpoint

Functions:
  * run_training_pipeline -- prepare resources, run the lifecycle, then method extensions
  * train_and_save -- train/evaluate and save the standard run artifacts

MLM extends ExternalLossTrainer; AR/S2S currently use their models' own losses
through the ordinary Hugging Face Trainer.
"""
import logging
from pathlib import Path

from transformers import Trainer, TrainerCallback

from cdrseqgen.methods.common.record_results import plot_training_curves, save_training_csv

logger = logging.getLogger(__name__)


class ExternalLossTrainer(Trainer):
    """Trainer for a method that calculates loss outside model.forward().

    Workflow:
      batch -> separate labels and loss metadata from model inputs
            -> model(**model_inputs)
            -> loss_from_outputs(outputs, labels, batch)
            -> scalar loss, optionally paired with the original outputs

    Subclasses declare LOSS_METADATA_KEYS and implement loss_from_outputs().
    The batch must already contain all inputs needed for the model forward.
    Losses are microbatch means; num_items_in_batch is deliberately unused.
    """

    LOSS_METADATA_KEYS = frozenset()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The external objective handles its own reduction, rather than using
        # Trainer's optional accumulated-token denominator.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if "labels" not in inputs:
            raise ValueError("training requires labels")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels" and key not in self.LOSS_METADATA_KEYS
        }
        outputs = model(**model_inputs)
        loss = self.loss_from_outputs(outputs, inputs["labels"], inputs)
        return (loss, outputs) if return_outputs else loss

    def loss_from_outputs(self, outputs, labels, inputs):
        """Return the method's scalar objective from outputs and batch metadata."""
        raise NotImplementedError("subclasses must implement loss_from_outputs()")


class SaveTokenizerCallback(TrainerCallback):
    """Each intermediate checkpoint carries the exact vocabulary used to train it."""
    def __init__(self, save_tokenizer):
        self.save_tokenizer = save_tokenizer

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            self.save_tokenizer(str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))
        return control


def run_training_pipeline(model_args, data_args, gen_args, training_args, *,
                          prepare_training, finish_training):
    """Run method preparation, shared training/saving and method extensions.

    prepare_training -> (Trainer, tokenizer)
      -> train_and_save -> finish_training -> return Trainer

    Extensions receive the Trainer after checkpoint selection and saving. They
    also run for evaluation-only commands and decide which optional stages apply.
    Exceptions propagate; a failed stage prevents subsequent stages from running.
    """
    trainer, tokenizer = prepare_training(model_args, data_args, training_args)
    train_and_save(trainer, training_args, tokenizer=tokenizer)
    finish_training(trainer, tokenizer, data_args, gen_args, training_args)
    return trainer


def train_and_save(trainer, training_args, *, tokenizer=None):
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
