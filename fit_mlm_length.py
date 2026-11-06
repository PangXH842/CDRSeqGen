"""Fit or refit the optional length head for an existing MLM checkpoint.

To run:
  python fit_mlm_length.py --model_dir <mlm_run>
  python fit_mlm_length.py --model_dir <mlm_run> --epochs 20 --device cpu

Functions:
  * build_parser / parse_args -- checkpoint, fitting overrides and device
  * resolve_config_path      -- explicit config or the run's saved configuration
  * main                     -- validate -> load checkpoint/data -> fit the length head

Workflow:
  1. Read configuration; validate MLM identity and fitting options.
  2. Restore the checkpoint tokenizer/model and prepare training/evaluation data.
  3. mlm.length.fit_length_predictor() fits and saves the head; return it.

Notes:
  * MLM training normally fits this head in finish_training() when enabled.
    This command repairs missing artifacts or deliberately refits an existing head.
  * The sequence model stays frozen; mlm.length.fit_length_predictor owns fitting.
    Existing length_config.json and length_head.safetensors are replaced by a refit.
  * Bounded/inferred MLM generation needs the head. Exact/native lengths and
    current AR/S2S methods do not need it.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.artifacts import RUN_CONFIG_FILENAMES, find_run_config, read_run_config
from cdrseqgen.methods.common.arguments import parse_config_dict
from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.mlm.arguments import DataArguments, GenerationArguments, ModelArguments
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.length import fit_length_predictor
from cdrseqgen.methods.mlm.training import get_datasets
from cdrseqgen.methods.registry import checkpoint_method


def resolve_config_path(model_dir: str, config: str | None = None) -> Path:
    """Use an explicit config or discover the exact config saved with a run."""
    if config is not None:
        path = Path(config)
        if not path.is_file():
            raise FileNotFoundError(f"training config not found: {path}")
        return path.resolve()

    path = find_run_config(model_dir)
    if path is None:
        expected = ", ".join(RUN_CONFIG_FILENAMES)
        raise FileNotFoundError(
            f"no saved training config found in {model_dir} (expected one of: {expected}); "
            "pass --config explicitly"
        )
    return path.resolve()


def build_parser():
    """Return options for fitting a head beside an existing MLM sequence checkpoint."""
    parser = argparse.ArgumentParser(description="Fit or refit an MLM checkpoint's length predictor.")
    parser.add_argument(
        "--config",
        help="Training config JSON (default: discover it in --model_dir)",
    )
    parser.add_argument("--model_dir", required=True, help="Finished sequence checkpoint")
    parser.add_argument("--epochs", type=int, default=None, help="Head-fitting epochs; default from config")
    parser.add_argument("--learning_rate", type=float, default=None, help="Head learning rate; default from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Head batch size; default from config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def parse_args(argv=None):
    """Parse argv, or sys.argv[1:] when omitted."""
    return build_parser().parse_args(argv)


def main(args=None):
    """Return the fitted length predictor; leave sequence-model weights unchanged.

    Discover/parse configuration -> validate MLM identity and fitting settings
      -> load datasets with the saved tokenizer -> restore model -> fit/save head.
    Invalid fitting settings fail before datasets or model weights are loaded.
    """
    args = parse_args() if args is None else args
    configure_logging()
    config_path = resolve_config_path(args.model_dir, args.config)
    with config_path.open() as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("length fitting configuration must contain a JSON object")
    checkpoint_method(config, "mlm")
    saved_config = read_run_config(args.model_dir)
    if saved_config is not None:
        checkpoint_method(saved_config, "mlm")

    hf_parser = HfArgumentParser(
        (ModelArguments, DataArguments, GenerationArguments, TrainingArguments)
    )
    print(f"Using training config: {config_path}")
    model_args, data_args, _gen_args, training_args = parse_config_dict(
        hf_parser, config, allow_extra_keys=True)
    fitting = {
        "epochs": data_args.length_predictor_epochs if args.epochs is None else args.epochs,
        "learning_rate": (data_args.length_predictor_learning_rate
                          if args.learning_rate is None else args.learning_rate),
        "batch_size": data_args.length_predictor_batch_size if args.batch_size is None else args.batch_size,
    }
    for name in ("epochs", "batch_size"):
        value = fitting[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    rate = fitting["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    training_args.output_dir = args.model_dir
    training_args.report_to = []
    training_args.eval_strategy = "no"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else torch.device(args.device)

    backbone = resolve(model_args.model_type)
    train_dataset, eval_dataset, tokenizer = get_datasets(
        model_args, data_args, training_args, backbone, checkpoint_path=args.model_dir
    )
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)
    return fit_length_predictor(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        output_dir=args.model_dir,
        device=device,
        **fitting,
        seed=training_args.seed,
    )


if __name__ == "__main__":
    main()
