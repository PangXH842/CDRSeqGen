"""Coordinate method preparation, Trainer execution and saved run artifacts.

Classes:
  * ExternalLossTrainer -- forward model inputs, then call a method's loss hook
  * SaveTokenizerCallback -- save the tokenizer with each training checkpoint

Functions:
  * run_training_pipeline -- prepare -> train/evaluate/save -> method finishing steps
  * train_and_save -- run the enabled Trainer stages and export metrics/curves

Training workflow:
  1. train.py -> <method>.train_cli.main(): configure logging and parse arguments.
  2. The CLI passes those arguments and <method>.training functions here:
       prepare_training -> (Trainer, tokenizer)
       -> train_and_save -> finish_training -> return Trainer.
  3. finish_training may fit method-specific state and evaluate generated candidates
     through common.evaluation.post_training_generation().

Notes:
  * Hugging Face Trainer owns optimization, validation during training and checkpoint
    selection. This module coordinates calls and saves their outputs.
  * Trainer validation scores predictions against dataset labels. Generated-candidate
    evaluation is a separate, optional stage in the method's finish_training().
  * MLM extends ExternalLossTrainer; AR/S2S currently use their models' own losses
    through the ordinary Hugging Face Trainer.
"""
import logging
from pathlib import Path

from transformers import Trainer, TrainerCallback

from cdrseqgen.visualization.training_results import plot_training_curves, save_training_csv

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
        """Separate loss-only fields, run the model and delegate the scalar objective.

        return_outputs=True also returns the original model outputs to Trainer.
        """
        del num_items_in_batch
        if "labels" not in inputs:
            raise ValueError("training requires labels")
        # 1. Keep labels and method-declared loss metadata out of model.forward().
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels" and key not in self.LOSS_METADATA_KEYS
        }
        # 2. Run the model; the subclass interprets its outputs and batch metadata.
        outputs = model(**model_inputs)
        loss = self.loss_from_outputs(outputs, inputs["labels"], inputs)
        return (loss, outputs) if return_outputs else loss

    def loss_from_outputs(self, outputs, labels, inputs):
        """Return the method's scalar objective from outputs and batch metadata."""
        raise NotImplementedError("subclasses must implement loss_from_outputs()")


class SaveTokenizerCallback(TrainerCallback):
    """Save the tokenizer beside each checkpoint when Trainer emits on_save.

    Receives a tokenizer's save_pretrained callable. args.should_save controls
    which process writes checkpoint-<step>; train_and_save saves the final copy.
    """

    def __init__(self, save_tokenizer):
        self.save_tokenizer = save_tokenizer

    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            self.save_tokenizer(str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))
        return control


def run_training_pipeline(model_args, data_args, gen_args, training_args, *,
                          prepare_training, finish_training):
    """Run the shared workflow with method-supplied functions; return its Trainer.

    General pipeline:
      1. prepare_training(model_args, data_args, training_args)
           -> (Trainer, tokenizer), with model, datasets, metrics and callbacks ready.
      2. train_and_save(trainer, training_args, tokenizer=tokenizer)
           -> enabled training/validation stages and standard run artifacts.
      3. finish_training(trainer, tokenizer, data_args, gen_args, training_args)
           -> optional method stages, such as length fitting or candidate evaluation.

    The method's train_cli supplies already parsed arguments and both functions
    from its training.py. finish_training also runs for evaluation-only commands;
    the method checks which optional stages apply. Exceptions propagate, stopping
    subsequent stages.
    """
    # ------ 1. Let the method build its Trainer and tokenizer. ------
    trainer, tokenizer = prepare_training(model_args, data_args, training_args)

    # ------ 2. Run the enabled Trainer stages and save their artifacts. ------
    train_and_save(trainer, training_args, tokenizer=tokenizer)

    # ------ 3. Run method extensions on the selected or already loaded model. ------
    finish_training(trainer, tokenizer, data_args, gen_args, training_args)
    return trainer


def train_and_save(trainer, training_args, *, tokenizer=None):
    """Run enabled Trainer stages, save their outputs and export validation curves.

    Steps:
      1. do_train -> train -> save model, train_results.json and optional tokenizer.
      2. do_eval  -> validate -> log metrics and save eval_results.json.
      3. Export epoch_metrics.csv and curve PNGs from Trainer's validation history.

    The two switches are independent. Trainer handles optimization, validation
    during training and checkpoint selection; load_best_model_at_end restores the
    selected model before the final save. Generated-candidate evaluation happens
    later in finish_training(). Tokenizer copies preserve added tokens/sentinels
    beside intermediate checkpoints and the final model.
    """
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)

    # ------ 1. Train, then save the model, training metrics and tokenizer. ------
    if training_args.do_train:
        if callable(save_tokenizer):
            trainer.add_callback(SaveTokenizerCallback(save_tokenizer))
        logger.info("🚀 Starting Training...")
        train_result = trainer.train()
        trainer.save_model(training_args.output_dir)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

        if callable(save_tokenizer):
            save_tokenizer(training_args.output_dir)

    # ------ 2. Validate the current model against dataset labels and save metrics. ------
    if training_args.do_eval:
        logger.info("🔍 Starting Evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ------ 3. Export the recorded validation trajectory as CSV and plots. ------
    save_training_csv(trainer, training_args.output_dir)
    plot_training_curves(training_args.output_dir)
