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
import hashlib
import os
import json
import argparse
import re
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainingArguments

from cdrseqgen.artifacts import (
    RUN_CONFIG_FILENAME,
    file_identity,
    find_run_config,
    git_state,
    processed_metadata_path,
    read_run_config,
)
from cdrseqgen.methods.registry import resolve as resolve_method, checkpoint_method
from cdrseqgen.train.logging import configure_logging

# Entrypoints configure their own logging. This used to arrive for free by importing train.py, which
# ran basicConfig at import — an accident that broke the moment train.py was split.
configure_logging()
from cdrseqgen.eval.generative import build_judge


COMPATIBILITY_FIELDS = (
    ("target_region", "target CDR"),
    ("model_type", "backbone/tokenizer"),
    ("include_partner_chain", "partner-chain context"),
    ("include_antigen", "antigen context"),
    ("data_path", "processed dataset"),
    ("max_len", "maximum input length"),
    ("masking_policy", "training masking policy"),
)


def _normalise_compatibility_value(key, value):
    if key in {"include_partner_chain", "include_antigen"}:
        return bool(value)
    if key == "data_path" and value:
        return str(Path(value).expanduser().resolve())
    if key in {"target_region", "masking_policy"} and isinstance(value, str):
        return value.strip().lower()
    return value


def compatibility_problems(run_config, model_args, data_args):
    """Return checkpoint/config mismatches that would change model inputs or interpretation."""
    if not run_config:
        return []
    requested = {
        "target_region": data_args.target_region,
        "model_type": model_args.model_type,
        "include_partner_chain": data_args.include_partner_chain,
        "include_antigen": data_args.include_antigen,
        "data_path": data_args.data_path,
        "max_len": data_args.max_len,
        "masking_policy": getattr(data_args, "masking_policy", None),
    }
    problems = []
    for key, label in COMPATIBILITY_FIELDS:
        # Old checkpoints do not carry every newer field. Absence is unknown, not disagreement.
        if key not in run_config:
            continue
        trained = _normalise_compatibility_value(key, run_config.get(key))
        supplied = _normalise_compatibility_value(key, requested[key])
        if trained != supplied:
            problems.append(
                f"{label}: checkpoint recorded {run_config.get(key)!r}, "
                f"but --config supplies {requested[key]!r}"
            )
    return problems



def evaluation_provenance(config_path, model_dir, data_path):
    """Hashes needed to reconstruct which code, checkpoint, config, and data produced an eval."""
    model_dir = Path(model_dir)
    run_config_path = find_run_config(model_dir)
    weights = []
    seen = set()
    for pattern in ("model.safetensors", "model-*.safetensors", "pytorch_model.bin"):
        for path in sorted(model_dir.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            weights.append(file_identity(path))

    data_path = Path(data_path).expanduser()
    metadata_path = processed_metadata_path(data_path)
    dataset = {
        "path": str(data_path.expanduser().resolve()),
        "size_bytes": data_path.stat().st_size if data_path.is_file() else None,
        "metadata": file_identity(metadata_path),
    }
    if metadata_path.is_file():
        with metadata_path.open() as handle:
            dataset["declared_processed_file_sha256"] = json.load(handle).get(
                "processed_file_sha256"
            )

    repo = Path(__file__).resolve().parent
    return {
        "config": file_identity(config_path),
        "run_config": file_identity(run_config_path),
        "checkpoint_weights": weights,
        "dataset": dataset,
        **git_state(repo),
    }


def default_output_path(args):
    """Parameter-addressed manual-eval path; a repeated identical run requires --overwrite."""
    def slug(value):
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")

    identity = {
        "length_mode": args.length_mode,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "length_temperature": args.length_temperature,
        "decoding": args.decoding,
        "num_examples": args.num_examples,
        "stratify": args.stratify,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "skip_perplexity": args.skip_perplexity,
        "judge": args.judge,
        "judge_path": args.judge_path,
    }
    if getattr(args, 'max_new_tokens', None) is not None:
        identity['max_new_tokens'] = args.max_new_tokens
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    judge = slug(args.judge or "nojudge").replace("/", "-")
    name = (
        f"{args.length_mode}_{args.decoding}_"
        f"n{args.num_examples}x{args.num_samples}_"
        f"T{slug(args.temperature)}_k{args.top_k}_p{slug(args.top_p)}_"
        f"s{args.seed}_{judge}_{digest}.json"
    )
    return os.path.join(args.model_dir, "evaluations", name)



def main():
    ap = argparse.ArgumentParser(description="Sample + score CDR designs over the held-out eval split.")
    ap.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    ap.add_argument("--model_dir", required=True, help="Run dir holding the finetuned model.")
    ap.add_argument("--num_examples", type=int, default=200, help="Eval examples to sample (-1 = all).")
    ap.add_argument("--stratify", action="store_true", help="Balance the draw across H3 length buckets.")

    ap.add_argument("--decoding", choices=["independent", "iterative", "causal"], default=None,
                    help="Default: independent for full_span checkpoints; a mixed checkpoint uses "
                         "its recorded decoder, or iterative when none was recorded.")
    ap.add_argument("--num_samples", type=int, default=20, help="Candidates per framework.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--length_mode", choices=("fixed_native", "bounded", "inferred"),
                    default=None)
    ap.add_argument("--min_length", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=None)
    ap.add_argument("--length_temperature", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=None, help="AR decode cap, including EOS.")
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--skip_perplexity", action="store_true", help="Skip self pseudo-perplexity.")
    ap.add_argument("--judge", type=str, default=None,
                    help="model_type of an INDEPENDENT LM to score every candidate "
                         "(makes perplexity comparable ACROSS models; self-ppl is not).")
    ap.add_argument("--judge_path", type=str, default=None, help="Weights for the judge (default: the --judge hub id).")
    ap.add_argument("--output", type=str, default=None,
                    help="Results JSON path ('none' to skip writing; default: a parameter-addressed "
                         "file under <model_dir>/evaluations/).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace an existing output with the same path/parameters.")
    ap.add_argument("--force", action="store_true",
                    help="Evaluate despite checkpoint/config input-contract mismatches. Every "
                         "mismatch is retained in the output provenance.")
    args = ap.parse_args()
    with open(args.config) as handle:
        supplied_config = json.load(handle)
    run_config = read_run_config(args.model_dir)
    method = resolve_method(supplied_config.get('method', 'mlm'))
    if run_config is not None:
        checkpoint_method(run_config, method.name)  # architecture mismatches cannot be forced
    cli = method.module('cli')
    cli.prepare_evaluation(args, run_config or supplied_config)
    argument_module = method.module('arguments')
    parser = HfArgumentParser((argument_module.ModelArguments, argument_module.DataArguments,
                              argument_module.GenerationArguments, TrainingArguments))
    model_args, data_args, _gen_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config))
    training_args.output_dir = args.model_dir
    training_args.report_to = []
    training_args.eval_strategy = "no"

    problems = compatibility_problems(run_config, model_args, data_args)
    if run_config is None:
        print(f"ℹ️  No {RUN_CONFIG_FILENAME} in {args.model_dir} (older run) — "
              "checkpoint/config compatibility cannot be verified.")
    elif problems:
        for problem in problems:
            print(f"⛔ {problem}")
        if not args.force:
            raise SystemExit(
                f"Refusing to evaluate: {len(problems)} checkpoint/config mismatch(es). "
                "Use the checkpoint's original input contract, or pass --force for a deliberate "
                "off-contract experiment."
            )
        print(f"🔓 --force given: continuing with {len(problems)} recorded mismatch(es).")
    else:
        print("🧬 Checkpoint input contract matches --config.")

    trained_for = run_config.get("target_region") if run_config else None
    output_path = None if args.output == "none" else (args.output or default_output_path(args))
    if output_path and os.path.exists(output_path) and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing evaluation: {output_path}. "
            "Choose --output or pass --overwrite."
        )
    provenance = evaluation_provenance(args.config, args.model_dir, data_args.data_path)

    backbone = method.module('backbones').resolve(model_args.model_type)
    _, eval_dataset, tokenizer = method.module('training').get_datasets(model_args, data_args, training_args, backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)

    method_options = cli.evaluation_options(args, device)

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

    method.evaluate(
        model, tokenizer, eval_dataset,
        device=device,
        num_samples=args.num_samples,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        **method_options,
        num_examples=args.num_examples, stratify=args.stratify, seed=args.seed,
        skip_perplexity=args.skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        judge_dataset=judge_dataset, judge_name=args.judge,
        anchor_trim=model_args.anchor_trim,
        output_path=output_path,
        title=f"{os.path.basename(args.model_dir)}   target={data_args.target_region}",
        meta={
            "config": str(Path(args.config).resolve()),
            "model_dir": str(Path(args.model_dir).resolve()),
            "target_region": data_args.target_region,        # loop this eval ran on (from --config)
            "model_trained_for": trained_for,                # loop the checkpoint was trained for (run_config)
            "eval_split": data_args.eval_split,
            "include_partner_chain": bool(data_args.include_partner_chain),
            "include_antigen": bool(data_args.include_antigen),
            "compatibility": {
                "problems": problems,
                "forced": bool(problems and args.force),
            },
            "provenance": provenance,
        },
    )


if __name__ == "__main__":
    main()
