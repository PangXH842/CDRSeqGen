"""Fit a gap-sentinel length predictor for an existing masked-LM checkpoint.

This is the migration path for checkpoints trained before length artifacts were
written automatically by ``train.py``. The sequence-model weights stay frozen.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.artifacts import RUN_CONFIG_FILENAMES
from cdrseqgen.methods.mlm.arguments import DataArguments, GenerationArguments, ModelArguments
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.length_predictor import fit_length_predictor
from cdrseqgen.methods.mlm.training import get_datasets


def resolve_config_path(model_dir: str, config: str | None = None) -> Path:
    """Use an explicit config or discover the exact config saved with a run."""
    if config is not None:
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"training config not found: {path}")
        return path.resolve()

    checkpoint = Path(model_dir)
    for filename in RUN_CONFIG_FILENAMES:
        path = checkpoint / filename
        if path.is_file():
            return path.resolve()
    expected = ", ".join(RUN_CONFIG_FILENAMES)
    raise FileNotFoundError(
        f"no saved training config found in {checkpoint} (expected one of: {expected}); "
        "pass --config explicitly"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        help="Training config JSON (default: discover it in --model_dir)",
    )
    parser.add_argument("--model_dir", required=True, help="Finished sequence checkpoint")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    hf_parser = HfArgumentParser(
        (ModelArguments, DataArguments, GenerationArguments, TrainingArguments)
    )
    config_path = resolve_config_path(args.model_dir, args.config)
    print(f"Using training config: {config_path}")
    model_args, data_args, _gen_args, training_args = hf_parser.parse_json_file(
        json_file=os.path.abspath(config_path),
        allow_extra_keys=True,
    )
    training_args.output_dir = args.model_dir
    training_args.report_to = []
    training_args.eval_strategy = "no"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone
    )
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)
    fit_length_predictor(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir=args.model_dir,
        device=device,
        epochs=args.epochs or data_args.length_predictor_epochs,
        learning_rate=args.learning_rate or data_args.length_predictor_learning_rate,
        batch_size=args.batch_size or data_args.length_predictor_batch_size,
        seed=training_args.seed,
    )


if __name__ == "__main__":
    main()
