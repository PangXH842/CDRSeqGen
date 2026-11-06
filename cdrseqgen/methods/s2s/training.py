"""Prepare S2S training resources and run method-specific finishing stages.

Entry path:
  train_cli.main -> common.training.run_training_pipeline
                 -> prepare_training -> common.training.train_and_save
                 -> finish_training

Classes/Functions:
  * ConditioningProbe        -- optional encoder-conditioning measurements and diagnostic stop
  * prepare_training         -- seed, load data/model, build the Trainer, attach diagnostics
  * finish_training          -- optionally evaluate generated loops on the selected/loaded model
  * get_datasets             -- return training/evaluation subsets and their tokenizer
  * compute_metrics          -- calculate loop recovery from already-aligned predictions
  * build_trainer            -- connect the model, collator, recovery metric and callbacks

Method details:
  * train_cli owns logging and argument setup; common.training owns train/eval/save.
  * Named split selection is shared; evaluation disables training context dropout.
  * The dataset offsets decoder inputs and labels; recovery must not shift them again.
  * Encoder and decoder padding may differ. The model supplies its loss, with an
    optional compute_loss_func override; ConditioningProbe is attached when requested.
"""
import functools
import json
import logging
import random
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, set_seed

from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset, s2s_collate_fn
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common import evaluation as shared_evaluation
from cdrseqgen.methods.common.eos_decoding import eos_recovery_metrics, get_infilling_datasets
from cdrseqgen.methods.common.run_setup import update_latest_link
from cdrseqgen.methods.s2s.backbones import resolve

logger = logging.getLogger(__name__)


def prepare_training(model_args, data_args, training_args):
    """Return (Trainer, tokenizer) ready for the shared training lifecycle.

    Steps:
      seed -> backbone -> datasets/tokenizer -> update latest link
           -> model -> Trainer/callbacks -> optional conditioning probe

    Validate dataset/split selection before updating the latest link. The shared
    lifecycle starts training/evaluation after this function returns.
    """
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)
    # Keep dataset/split failures from changing the latest-run link.
    update_latest_link(training_args.output_dir)

    model = backbone.load_training_model(
        model_args.model_type,
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_layers=model_args.num_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_len=data_args.max_len,
        num_encoder_layers=model_args.num_encoder_layers,
        num_decoder_layers=model_args.num_decoder_layers,
        encoder_init=model_args.encoder_init,
        freeze_encoder=model_args.freeze_encoder,
    )
    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                            callbacks=build_callbacks(model_args, training_args, logger),
                            trim=model_args.anchor_trim)

    if model_args.conditioning_probe_steps or model_args.diagnostic_stop_epoch:
        trainer.add_callback(ConditioningProbe(
            eval_dataset, trainer.data_collator, model_args.conditioning_probe_steps,
            model_args.diagnostic_stop_epoch))
    return trainer, tokenizer


def finish_training(trainer, tokenizer, data_args, gen_args, training_args, *, anchor_trim):
    """Optionally evaluate generated loops on the selected or loaded model.

    Also runs for evaluation-only commands. train_cli binds anchor_trim from
    model_args; run_generative_eval controls this stage independently of do_train.
    Pass the S2S decode cap directly to common.evaluation.post_training_generation().
    """
    if gen_args.run_generative_eval:
        shared_evaluation.post_training_generation(
            "s2s", trainer, tokenizer, trainer.eval_dataset,
            data_args, gen_args, training_args,
            anchor_trim=anchor_trim,
            method_options={"max_new_tokens": gen_args.gen_max_new_tokens},
        )


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None):
    """Return (train subset, evaluation subset, tokenizer) through the shared data helper.

    Both views share parsed records; evaluation disables context dropout.
    checkpoint_path selects a saved tokenizer for standalone checkpoint evaluation.
    """
    return get_infilling_datasets(Seq2SeqInfillingDataset, "seq2seq", model_args, data_args,
                                  training_args, backbone, checkpoint_path=checkpoint_path)


def compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Return teacher-forced whole/core/anchor recovery from aligned decoder predictions.

    Example: loop A C
      decoder input: BOS A C
      labels:         A  C EOS
      predictions:    A  C EOS

    Accept logits or already-decoded token IDs. Compare predictions and labels
    at the same position; eos_recovery_metrics drops the final supervised terminator.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    return eos_recovery_metrics(predictions, labels, trim=trim)


def build_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                  trim=ANCHOR_TRIM, compute_loss_func=None):
    """Return an HF Trainer with S2S collation, recovery metrics and callbacks.

    Encoder padding comes from encoder_tokenizer when present; decoder padding
    comes from tokenizer. The model calculates loss by default. An optional
    compute_loss_func receives aligned labels and follows
    f(outputs, labels, num_items_in_batch=None) -> scalar: divide summed loss by
    the supplied accumulation-window count, or return a token mean when absent.
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(
            s2s_collate_fn, pad_token_id=tokenizer.pad_id,
            encoder_pad_token_id=getattr(tokenizer, "encoder_tokenizer", tokenizer).pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=trim),
        compute_loss_func=compute_loss_func,
        callbacks=callbacks or [],
    )


class ConditioningProbe(TrainerCallback):
    """Probe S2S encoder conditioning during training, with optional diagnostic stop.

    Caller:
      prepare_training -> trainer.add_callback -> Trainer event methods

    Measurements:
      fixed validation batch -> normal encoder context vs row-swapped context
                             -> encoder spread, finite values and prediction changes
                             -> conditioning_probe.jsonl

    interval enables measurements at train start, every interval optimizer steps
    and each epoch end, on the main process. stop_epoch independently requests
    stopping, evaluation and saving without changing the planned LR schedule.
    Measurements preserve model mode and Torch RNG; they do not select checkpoints.
    """
    def __init__(self, dataset, collate, interval=250, stop_epoch=0):
        """Choose up to 16 fixed validation rows; collate them when probing is enabled."""
        self.interval = interval
        self.stop_epoch = stop_epoch
        indices = random.Random(42).sample(range(len(dataset)), min(16, len(dataset)))
        if interval and len(indices) < 2:
            raise ValueError("Conditioning probes require at least two validation examples")
        self.indices = indices
        self.batch = collate([dataset[i] for i in indices]) if interval else None

    def measure(self, args, state, model):
        """Append one conditioning record while preserving model mode and Torch RNG."""
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in self.batch.items()}
        was_training = model.training
        # Even future stochastic probe operations must not alter the training RNG stream.
        devices = [device.index] if device.type == "cuda" else []
        try:
            model.eval()
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                memory = model.encode(batch['input_ids'], batch['attention_mask'])
                valid = batch['attention_mask'].bool()
                original = model(**batch, encoder_hidden_states=memory)['logits']
                swapped = dict(batch, attention_mask=batch['attention_mask'].roll(1, 0))
                changed = model(**swapped, encoder_hidden_states=memory.roll(1, 0))['logits']
                targets = batch['labels'].ne(-100)
                row = dict(step=state.global_step, epoch=state.epoch,
                           validation_subset_indices=self.indices,
                           projected_token_std=memory[valid].float().std(0).mean().item(),
                           finite=bool(torch.isfinite(memory[valid]).all() and torch.isfinite(original[targets]).all()),
                           swap_mean_abs_logit_change=(original-changed).abs()[targets].mean().item(),
                           swap_argmax_change_rate=(original.argmax(-1)[targets]
                                                    != changed.argmax(-1)[targets]).float().mean().item())
        finally:
            model.train(was_training)
        with (Path(args.output_dir) / 'conditioning_probe.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Measure the initial model on the main process when probing is enabled."""
        if self.interval and state.is_world_process_zero:
            self.measure(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Measure periodically and request stop/evaluation/save at stop_epoch."""
        if self.interval and state.global_step % self.interval == 0 and state.is_world_process_zero:
            self.measure(args, state, model)
        if self.stop_epoch and state.epoch >= self.stop_epoch:
            control.should_training_stop = True
            control.should_evaluate = True
            control.should_save = True
        return control

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Measure after each epoch on the main process when probing is enabled."""
        if self.interval and state.is_world_process_zero:
            self.measure(args, state, model)
