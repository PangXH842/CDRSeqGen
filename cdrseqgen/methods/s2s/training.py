"""Encoder-decoder training pipeline: dataset -> Trainer -> metrics.

  * get_datasets           -- tokenizer + Seq2SeqInfillingDataset + the train/eval split
  * s2s_compute_metrics    -- teacher-forced recovery over the loop, split core/anchor
  * build_seq2seq_trainer  -- a plain HF Trainer; the model returns its own loss

Mirrors ``methods/ar/training.py`` so the two are comparable. Two differences, both forced:

  * **No shift in the metrics.** The causal side receives predictions and labels aligned to the same
    sequence and must redo the shift itself. Here ``dataset_s2s`` already offset the decoder pair,
    so the prediction at t is scored against the label at t directly. Copying the causal version
    verbatim would offset every position by one and quietly report a different quantity — which is
    the whole reason this docstring exists rather than a "same as AR" comment.
  * **The loop is the entire supervised stream.** There is no prompt to mask out with -100, so
    ``core_coverage`` is bounded only by loop length, not by whether a prefix was labelled.

Metrics are teacher-forced, like both siblings' ``compute_metrics``: argmax over the true prefix,
not free-running generation. Findings' "Teacher-forced accuracy is not generation accuracy" applies
here too, and the gap will be at least as large as the causal one, so these numbers select
checkpoints — they do not report design quality.
"""
import functools
import logging

import numpy as np
import torch
from transformers import Trainer

from cdrseqgen.datasets.dataset_common import select_split_indices, evaluation_context_view
from cdrseqgen.datasets.dataset_s2s import Seq2SeqInfillingDataset, s2s_collate_fn
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.metrics import core_anchor_indices

logger = logging.getLogger(__name__)


def get_datasets(model_args, data_args, training_args, backbone, *, checkpoint_path=None):
    """Tokenizer, train/eval subsets. Same signature as the masked and causal pipelines'."""
    tokenizer = backbone.build_tokenizer(model_args.model_type, checkpoint_path=checkpoint_path)
    full = Seq2SeqInfillingDataset(
        data_args.data_path,
        tokenizer,
        max_len=data_args.max_len,
        target_region=data_args.target_region,
        include_partner_chain=data_args.include_partner_chain,
        include_antigen=data_args.include_antigen,
        context_dropout=getattr(data_args, "context_dropout", 0.0),
        context_dropout_seed=training_args.seed,
    )
    train_idx, eval_idx, eval_label = select_split_indices(
        full.data, getattr(data_args, "eval_split", ""), data_path=data_args.data_path,
        seed=training_args.seed, test_size=getattr(data_args, "test_size", 0.1))

    # Share records while keeping training context dropout out of validation.
    train_ds = torch.utils.data.Subset(full, train_idx)
    eval_ds = torch.utils.data.Subset(evaluation_context_view(full), eval_idx)
    logger.info(f"📊 Data loaded: {len(train_ds)} train (seq2seq), "
                f"{len(eval_ds)} eval (split: {eval_label}).")
    return train_ds, eval_ds, tokenizer


def s2s_compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Teacher-forced recovery over the loop: whole-loop, plus anchor-excluded core and anchor.

    Predictions and labels arrive ALIGNED — position t of the prediction answers position t of the
    label, because the dataset built the decoder pair offset by one. Do not re-shift here.

    The final supervised position of each row is its <eos>; it is excluded from recovery so this
    number measures residue prediction, comparable to both siblings, rather than being diluted by a
    token the masked method has no equivalent of.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)

    whole_hits = whole_total = core_hits = core_total = anchor_hits = anchor_total = 0
    rows_with_core = 0
    for row_preds, row_gold in zip(predictions, labels):
        positions = np.flatnonzero(row_gold != -100)
        if positions.size == 0:
            continue
        residues = positions[:-1]                      # drop the trailing <eos>
        if residues.size == 0:
            continue
        correct = row_preds[residues] == row_gold[residues]
        whole_hits += int(correct.sum())
        whole_total += residues.size

        core, anchor = core_anchor_indices(residues.size, trim)
        if core:
            rows_with_core += 1
            core_hits += int(correct[core].sum())
            core_total += len(core)
        anchor_hits += int(correct[anchor].sum())
        anchor_total += len(anchor)

    rows = int(sum(1 for row in labels if (row != -100).any()))
    return {
        "accuracy": whole_hits / whole_total if whole_total else 0.0,
        "core_accuracy": core_hits / core_total if core_total else 0.0,
        "anchor_accuracy": anchor_hits / anchor_total if anchor_total else 0.0,
        "core_coverage": rows_with_core / rows if rows else 0.0,
    }


def build_seq2seq_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                          trim=ANCHOR_TRIM):
    """A plain HF Trainer: like the causal side there is no external loss to bridge."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(
            s2s_collate_fn, pad_token_id=tokenizer.pad_id,
            encoder_pad_token_id=getattr(tokenizer, "encoder_tokenizer", tokenizer).pad_id),
        compute_metrics=functools.partial(s2s_compute_metrics, trim=trim),
        callbacks=callbacks or [],
    )
