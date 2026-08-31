"""Generative evaluation of a finished checkpoint, on the held-out eval split.

Entrypoint used to evaluate trained model checkpoints, especially CDR generation capabilities.
Use ``--judge`` to choose the shared reference model explicitly; training-time evaluation uses its
configured ``gen_judge`` default.

Usage example:
    python evaluate.py --config configs/h3_esm2.json \
        --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
        --num_examples 150 --stratify --num_samples 20 --top_p 0.9 \
        --judge Exscientia/IgBert
"""
import os
import json
import argparse

import torch
from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.artifacts import RUN_CONFIG_FILENAME, load_run_config
from cdrseqgen.methods.mlm.arguments import DataArguments, GenerationArguments, ModelArguments
from cdrseqgen.methods.mlm.training import get_datasets
from cdrseqgen.train.logging import configure_logging

# Entrypoints configure their own logging. This used to arrive for free by importing train.py, which
# ran basicConfig at import — an accident that broke the moment train.py was split.
configure_logging()
from cdrseqgen.methods.mlm.backbones import resolve
from cdrseqgen.methods.mlm.evaluation import build_judge, run_generative_eval
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy


def read_run_config(model_dir):
    """Load the run_config.json a training run wrote, or None (older runs won't have it)."""
    try:
        config, _ = load_run_config(model_dir)
    except FileNotFoundError:
        return None
    return config


def resolve_decoding(requested, model_dir):
    """Pick the decoder, defaulting to the one the checkpoint was trained and reported under.

    A hardcoded default here is not a neutral choice. `iterative` re-masks and resamples positions
    one at a time, which assumes the model saw partially revealed loops in training — true only for
    a `mixed` checkpoint. Applied to the project's default `full_span` checkpoints it is off
    distribution and expensive: measured on h3_OAS_scratch over 300 frameworks x 10 candidates,
    iterative decoding drops whole-loop recovery 0.370 -> 0.253 and CORE recovery 0.204 -> 0.077,
    i.e. below the 0.084 you get by sampling H3's marginal amino-acid frequencies. So the default
    now follows the run rather than overriding it.
    """
    meta = read_run_config(model_dir) or {}
    recorded = meta.get("gen_decoding")
    policy = meta.get("masking_policy")
    if requested is None:
        chosen = recorded or "independent"
        source = f"{RUN_CONFIG_FILENAME}" if recorded else "fallback"
        print(f"🎲 Decoding: {chosen} (from {source})")
    else:
        chosen = requested
        if recorded and recorded != chosen:
            print(f"🎲 Decoding: {chosen} (CLI override; the run recorded '{recorded}')")
        else:
            print(f"🎲 Decoding: {chosen}")
    if chosen == "iterative" and policy == "full_span":
        print("⚠️  Iterative decoding on a full_span checkpoint: the model never saw a partially "
              "revealed loop in training, so this is off distribution and roughly halves core "
              "recovery. Train with masking_policy: mixed for iterative decoding.")
    return chosen


def check_target_region(model_dir, config_target):
    """Report the CDR loop the checkpoint was trained for, and warn if the config disagrees.

    The eval runs on config_target (the dataset is built from --config). If the checkpoint's own
    run_config says it was trained for a DIFFERENT loop, we're masking the wrong CDR — the numbers
    would be confident and meaningless — so warn loudly. Returns the model's trained loop (or None).
    """
    meta = read_run_config(model_dir)
    trained_for = meta.get("target_region") if meta else None
    if trained_for is None:
        print(f"ℹ️  No {RUN_CONFIG_FILENAME} in {model_dir} (older run) — can't verify the model's trained loop.")
    elif trained_for != config_target:
        print(f"⚠️  LOOP MISMATCH: model was trained for '{trained_for}', but --config targets "
              f"'{config_target}'. Evaluating on the WRONG CDR — metrics will be meaningless.")
    else:
        print(f"🧬 Model trained for CDR '{trained_for}' (matches config).")
    return trained_for


def main():
    ap = argparse.ArgumentParser(description="Sample + score CDR designs over the held-out eval split.")
    ap.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    ap.add_argument("--model_dir", required=True, help="Run dir holding the finetuned model.")
    ap.add_argument("--num_examples", type=int, default=200, help="Eval examples to sample (-1 = all).")
    ap.add_argument("--stratify", action="store_true", help="Balance the draw across H3 length buckets.")

    ap.add_argument("--decoding", choices=["independent", "iterative"], default=None,
                    help="Default: whatever the checkpoint's run_config.json recorded, so the "
                         "eval decoder matches the one the run was trained and reported under.")
    ap.add_argument("--num_samples", type=int, default=20, help="Candidates per framework.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--length_mode", choices=("fixed_native", "bounded", "inferred"),
                    default="fixed_native")
    ap.add_argument("--min_length", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--length_temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--skip_perplexity", action="store_true", help="Skip self pseudo-perplexity.")
    ap.add_argument("--judge", type=str, default=None,
                    help="model_type of an INDEPENDENT LM to score every candidate "
                         "(makes perplexity comparable ACROSS models; self-ppl is not).")
    ap.add_argument("--judge_path", type=str, default=None, help="Weights for the judge (default: the --judge hub id).")
    ap.add_argument("--output", type=str, default=None,
                    help="Results JSON path ('none' to skip writing; default: <model_dir>/generation_eval.json).")
    args = ap.parse_args()
    resolve_generation_length_policy(
        args.length_mode,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, _gen_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config))
    training_args.output_dir = args.model_dir
    training_args.report_to = []
    training_args.eval_strategy = "no"

    # Provenance: what CDR was this checkpoint actually trained for? (warns on config mismatch)
    trained_for = check_target_region(args.model_dir, data_args.target_region)
    args.decoding = resolve_decoding(args.decoding, args.model_dir)

    backbone = resolve(model_args.model_type)
    _, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)

    length_predictor = None
    if args.length_mode != "fixed_native":
        from cdrseqgen.methods.mlm.length_predictor import LengthPredictor
        length_predictor = LengthPredictor.load(args.model_dir, device=device)

    judge_model = judge_tokenizer = judge_dataset = None
    if args.judge:
        print(f"⚖️  Loading judge model: {args.judge}")
        judge_model, judge_tokenizer, judge_dataset = build_judge(
            args.judge, args.judge_path,
            data_path=data_args.data_path,
            max_len=data_args.max_len,
            target_region=data_args.target_region,
            include_partner_chain=data_args.include_partner_chain,
            include_antigen=data_args.include_antigen,
            device=device,
        )

    run_generative_eval(
        model, tokenizer, eval_dataset,
        device=device,
        decoding=args.decoding, num_samples=args.num_samples,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        length_mode=args.length_mode, length_predictor=length_predictor,
        min_length=args.min_length, max_length=args.max_length,
        length_temperature=args.length_temperature,
        num_examples=args.num_examples, stratify=args.stratify, seed=args.seed,
        skip_perplexity=args.skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        judge_dataset=judge_dataset, judge_name=args.judge,
        output_path=(None if args.output == "none"
                     else args.output or os.path.join(args.model_dir, "generation_eval.json")),
        title=f"{os.path.basename(args.model_dir)}   target={data_args.target_region}",
        meta={
            "config": args.config,
            "model_dir": args.model_dir,
            "target_region": data_args.target_region,        # loop this eval ran on (from --config)
            "model_trained_for": trained_for,                # loop the checkpoint was trained for (run_config)
            "include_partner_chain": bool(data_args.include_partner_chain),
            "include_antigen": bool(data_args.include_antigen),
        },
    )


if __name__ == "__main__":
    main()
