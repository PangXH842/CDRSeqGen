"""Re-evaluate a finished run's saved best model with the anchor-aware recovery metric.

Retrains nothing: reloads the finetuned checkpoint from --model_dir and re-runs eval on the same
deterministic eval split, reporting whole-span accuracy, anchor-excluded *core* accuracy (the
hypervariable middle that is actually being designed), anchor-only accuracy, and a
position-from-start breakdown. Data/model args are read from the original --config, so the eval
set is reconstructed identically (SAbDab2 uses its own 'split' field; OAS uses the same
seed/test_size).

Example:
    python rescore.py --config configs/h3_esm2_ag.json \
        --model_dir results/20260703_210913_cdr3_SD_LC_AG_esm2_t12_35M_UR50D
"""
import os
import json
import argparse
import functools

import numpy as np
from transformers import HfArgumentParser, TrainingArguments, Trainer

from train import (
    ModelArguments,
    DataArguments,
    get_datasets,
    compute_metrics,
    _core_anchor_masks,
    ANCHOR_TRIM,
)
from backbones import resolve
from dataset import collate_fn

MAX_POS = 16  # positions-from-start reported in the stratified breakdown


def main():
    ap = argparse.ArgumentParser(description="Re-score a finished run with the anchor-aware metric.")
    ap.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    ap.add_argument("--model_dir", required=True, help="Run dir holding the finetuned best model.")
    ap.add_argument("--json", action="store_true", help="Emit one JSON line instead of a table.")
    args = ap.parse_args()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config))
    training_args.output_dir = args.model_dir  # evaluate-only; nothing is written here
    training_args.report_to = []
    training_args.eval_strategy = "no"  # we drive eval manually via trainer.predict()

    backbone = resolve(model_args.model_type)
    _, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)
    model = backbone.load_pretrained_model(args.model_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=functools.partial(collate_fn, pad_token_id=tokenizer.pad_id),
        compute_metrics=compute_metrics,
    )

    pred = trainer.predict(eval_dataset)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    labels = pred.label_ids
    preds = np.argmax(logits, axis=-1)

    masked = labels != -100
    correct = (preds == labels) & masked
    core_mask, anchor_mask = _core_anchor_masks(masked)

    def acc(m):
        t = m.sum()
        return float((correct & m).sum() / t) if t else 0.0

    # Position-from-start stratified recovery (loops are variable length; align on the start).
    pos_correct = np.zeros(MAX_POS)
    pos_total = np.zeros(MAX_POS)
    for r in range(masked.shape[0]):
        cols = np.nonzero(masked[r])[0]
        for j, c in enumerate(cols[:MAX_POS]):
            pos_total[j] += 1
            pos_correct[j] += correct[r, c]
    pos_acc = [float(pos_correct[j] / pos_total[j]) if pos_total[j] else None for j in range(MAX_POS)]

    result = {
        "model_dir": args.model_dir,
        "target": data_args.target_region,
        "include_antigen": bool(data_args.include_antigen),
        "n_eval": int(masked.shape[0]),
        "anchor_trim": ANCHOR_TRIM,
        "accuracy": acc(masked),
        "core_accuracy": acc(core_mask),
        "anchor_accuracy": acc(anchor_mask),
        "pos_from_start_acc": pos_acc,
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\n📄 {os.path.basename(args.model_dir)}")
        print(f"   target={result['target']}  antigen={result['include_antigen']}  n_eval={result['n_eval']}  trim={ANCHOR_TRIM}")
        print(f"   whole-span accuracy : {result['accuracy']:.4f}")
        print(f"   CORE accuracy       : {result['core_accuracy']:.4f}   <-- anchor-excluded, the designable middle")
        print(f"   anchor accuracy     : {result['anchor_accuracy']:.4f}")
        shown = " ".join(f"{a:.2f}" if a is not None else "  -" for a in pos_acc if a is not None)
        print(f"   pos-from-start acc  : {shown}")


if __name__ == "__main__":
    main()
