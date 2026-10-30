"""Causal training pipeline: dataset -> Trainer -> metrics, and the hooks of the shared workflow.

Entry path:
  train_cli.main -> common.training.run_training_pipeline
                 -> prepare_training -> common.training.train_and_save
                 -> finish_training

  * prepare_training         -- seed, load data/model, build the Trainer
  * finish_training          -- evaluate generated loops after training
  * get_datasets             -- tokenizer + CausalInfillingDataset + the train/eval split
  * compute_metrics          -- teacher-forced recovery over the answer, split core/anchor
  * build_trainer            -- a plain HF Trainer; the model returns its own loss
  * post_training_generation -- pass the decode cap into common.evaluation

Mirrors ``methods/mlm/training.py`` so the two are comparable, with three differences that follow
from the objective rather than from taste:

  * **No masking view.** The MLM builds a separate ``full_span`` view for evaluation so partially
    revealed native residues cannot leak into reported recovery. A causal example never reveals any
    of the answer — the loss mask puts -100 across the whole prompt — so train and eval share parsed
    records, with a separate context view that disables dropout for evaluation.
  * **No loss registry.** The model computes its own next-token loss. §13 established that
    re-weighting core against anchor moves nothing, so there is no reason to rebuild that machinery
    on this side.
  * **The split guard is imported, not copied** (``dataset_common.select_split_indices``). Two
    copies of the check that stops checkpoints being selected on the test partition is precisely the
    duplication this project has already been burned by.

The dataset split, the recovery arithmetic and the evaluation loop are shared with the
encoder-decoder method in :mod:`cdrseqgen.methods.common.eos_decoding`; this module keeps what is
causal: the dataset and collator, and the one-position shift in ``compute_metrics``.

Metrics are teacher-forced, like the MLM's ``compute_metrics``: argmax over the true prefix, not
free-running generation. Findings' "Teacher-forced accuracy is not generation accuracy" applies here
too and the gap will be at least as large, so these numbers select checkpoints — they do not report
design quality.
"""
import functools
import logging

from transformers import Trainer, set_seed

from cdrseqgen.datasets.dataset_ar import CausalInfillingDataset, causal_collate_fn
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.methods.ar.backbones import record_training_layout, resolve
from cdrseqgen.methods.common import build_callbacks
from cdrseqgen.methods.common.eos_decoding import eos_recovery_metrics, get_infilling_datasets
from cdrseqgen.methods.common.run_setup import update_latest_link

logger = logging.getLogger(__name__)


def prepare_training(model_args, data_args, training_args):
    """Return (Trainer, tokenizer) ready for the shared training lifecycle.

    Steps:
      seed -> backbone -> datasets/tokenizer -> update latest link -> model -> Trainer/callbacks

    Dataset/split validation precedes the latest-link update. Training starts when the shared
    workflow calls train_and_save().
    """
    set_seed(training_args.seed)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone)
    # Only now: everything above can succeed for a config get_datasets then rejects.
    update_latest_link(training_args.output_dir)

    model = backbone.load_training_model(
        model_args.model_type,
        vocab_size=tokenizer.vocab_size,
        hidden_size=model_args.hidden_size,
        num_layers=model_args.num_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        max_len=data_args.max_len,
    )
    record_training_layout(model, tokenizer, training_args.output_dir)
    trainer = build_trainer(model, training_args, train_dataset, eval_dataset, tokenizer,
                            callbacks=build_callbacks(model_args, training_args, logger),
                            trim=model_args.anchor_trim)
    return trainer, tokenizer


def finish_training(trainer, tokenizer, data_args, gen_args, training_args, *, anchor_trim):
    """Evaluate generated loops on the trained model when run_generative_eval is on.

    anchor_trim is model_args.anchor_trim; train_cli binds it because the shared workflow does not
    pass model_args to this hook.
    """
    if gen_args.run_generative_eval:
        post_training_generation(trainer, tokenizer, trainer.eval_dataset, data_args, gen_args,
                                 training_args, anchor_trim=anchor_trim)


def post_training_generation(trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                             *, anchor_trim):
    """Pass the decode cap into shared post-training evaluation.

    common.evaluation handles judge setup, context regimes, output paths and exception logging.
    """
    from cdrseqgen.methods.common.evaluation import post_training_generation as evaluate
    return evaluate('ar', trainer, tokenizer, eval_dataset, data_args, gen_args, training_args,
                    anchor_trim=anchor_trim,
                    method_options={'max_new_tokens': gen_args.gen_max_new_tokens})


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None):
    """Tokenizer, train/eval subsets. Same signature as the MLM pipeline's."""
    return get_infilling_datasets(CausalInfillingDataset, "causal", model_args, data_args,
                                  training_args, backbone, checkpoint_path=checkpoint_path)


def compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Teacher-forced recovery over the answer: whole-loop, plus anchor-excluded core and anchor.

    Predictions and labels arrive UNSHIFTED, exactly as the model saw them, so the shift is redone
    here: the logit at t is scored against the label at t+1. Getting that wrong offsets every
    position by one and would quietly report a different quantity than the MLM's ``compute_metrics``.
    The arithmetic itself is :func:`eos_recovery_metrics`, which also drops each row's final <eos>.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    return eos_recovery_metrics(predictions[:, :-1], labels[:, 1:], trim=trim)


def build_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                  trim=ANCHOR_TRIM):
    """A plain HF Trainer: unlike the masked side there is no external loss to bridge."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(causal_collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(compute_metrics, trim=trim),
        callbacks=callbacks or [],
    )
