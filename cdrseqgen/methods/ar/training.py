"""Causal training pipeline: dataset -> Trainer -> metrics.

  * get_datasets            -- tokenizer + CausalInfillingDataset + the train/eval split
  * causal_compute_metrics  -- teacher-forced recovery over the answer, split core/anchor
  * build_causal_trainer    -- a plain HF Trainer; the model returns its own loss

Mirrors ``methods/mlm/training.py`` so the two are comparable, with three differences that follow
from the objective rather than from taste:

  * **No masking view.** The MLM builds a separate ``full_span`` view for evaluation so partially
    revealed native residues cannot leak into reported recovery. A causal example never reveals any
    of the answer — the loss mask puts -100 across the whole prompt — so train and eval share one
    dataset.
  * **No loss registry.** The model computes its own next-token loss. §13 established that
    re-weighting core against anchor moves nothing, so there is no reason to rebuild that machinery
    on this side.
  * **The split guard is imported, not copied** (``dataset_common.select_split_indices``). Two
    copies of the check that stops checkpoints being selected on the test partition is precisely the
    duplication this project has already been burned by.

Metrics are teacher-forced, like the MLM's ``compute_metrics``: argmax over the true prefix, not
free-running generation. Findings' "Teacher-forced accuracy is not generation accuracy" applies here
too and the gap will be at least as large, so these numbers select checkpoints — they do not report
design quality.
"""
import logging

import numpy as np
import torch
from transformers import Trainer

from cdrseqgen.data.dataset_ar import CausalInfillingDataset, causal_collate_fn
from cdrseqgen.data.dataset_common import select_split_indices
from cdrseqgen.eval.metrics import core_anchor_indices
from cdrseqgen.eval import ANCHOR_TRIM

logger = logging.getLogger(__name__)


def get_datasets(model_args, data_args, training_args, backbone):
    """Tokenizer, train/eval subsets. Same signature as the MLM pipeline's."""
    tokenizer = backbone.build_tokenizer(model_args.model_type)
    full = CausalInfillingDataset(
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

    # No masking view: a causal example never exposes the answer, so one dataset serves both.
    train_ds = torch.utils.data.Subset(full, train_idx)
    eval_ds = torch.utils.data.Subset(full, eval_idx)
    logger.info(f"📊 Data loaded: {len(train_ds)} train (causal), "
                f"{len(eval_ds)} eval (split: {eval_label}).")
    return train_ds, eval_ds, tokenizer


def causal_compute_metrics(eval_pred, *, trim=ANCHOR_TRIM):
    """Teacher-forced recovery over the answer: whole-loop, plus anchor-excluded core and anchor.

    Predictions and labels arrive UNSHIFTED, exactly as the model saw them, so the shift is redone
    here: the logit at t is scored against the label at t+1. Getting that wrong offsets every
    position by one and would quietly report a different quantity than the MLM's ``compute_metrics``.

    The final supervised position of each row is its <eos>; it is excluded from recovery so this
    number measures residue prediction, comparable to the masked side, rather than being diluted by
    a token the MLM has no equivalent of.
    """
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.argmax(-1)
    preds = predictions[:, :-1]
    gold = labels[:, 1:]

    whole_hits = whole_total = core_hits = core_total = anchor_hits = anchor_total = 0
    rows_with_core = 0
    for row_preds, row_gold in zip(preds, gold):
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

    rows = int(sum(1 for row in gold if (row != -100).any()))
    return {
        "accuracy": whole_hits / whole_total if whole_total else 0.0,
        "core_accuracy": core_hits / core_total if core_total else 0.0,
        "anchor_accuracy": anchor_hits / anchor_total if anchor_total else 0.0,
        "core_coverage": rows_with_core / rows if rows else 0.0,
    }


def build_causal_trainer(model, training_args, train_ds, eval_ds, tokenizer, callbacks=None,
                         trim=ANCHOR_TRIM):
    """A plain HF Trainer: unlike the masked side there is no external loss to bridge."""
    import functools
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=functools.partial(causal_collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=functools.partial(causal_compute_metrics, trim=trim),
        callbacks=callbacks or [],
    )
