"""Shared standalone and post-training evaluation lifecycles.

Functions:
  * run_evaluation_pipeline -- validate, load resources and call a method evaluator
  * post_training_generation -- evaluate trainer.model with optional context ablation
  * compatibility_problems -- compare checkpoint and requested input contracts
  * evaluation_provenance -- identify source/config/data/checkpoint artifacts
  * default_output_path -- derive the parameter-addressed standalone output path

Standalone errors propagate. Post-training evaluation logs its errors after
training artifacts have been saved. Candidate generation belongs to each method;
cdrseqgen.eval supplies measurements, scoring and reports.
"""
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import torch
from transformers import HfArgumentParser, TrainingArguments

import cdrseqgen
from cdrseqgen import artifacts
from cdrseqgen.artifacts import (
    RUN_CONFIG_FILENAME, file_identity, find_run_config, git_state,
    processed_metadata_path,
)
from cdrseqgen.datasets.dataset_common import evaluation_context_view
from cdrseqgen.eval.generative import build_judge
from cdrseqgen.methods.common.arguments import parse_config_dict
from cdrseqgen.methods.registry import resolve

logger = logging.getLogger(__name__)


COMPATIBILITY_FIELDS = (
    ("target_region", "target CDR"),
    ("model_type", "backbone/tokenizer"),
    ("include_partner_chain", "partner-chain context"),
    ("include_antigen", "antigen context"),
    ("data_path", "processed dataset"),
    ("max_len", "maximum input length"),
    ("masking_policy", "training masking policy"),
)


def _normalize_compatibility_value(key, value):
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
        trained = _normalize_compatibility_value(key, run_config.get(key))
        supplied = _normalize_compatibility_value(key, requested[key])
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

    repo = Path(cdrseqgen.__file__).resolve().parent.parent
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
    identity['partner_context'] = getattr(args, 'partner_context', 'trained')
    identity['judge_partner_context'] = getattr(args, 'judge_partner_context', 'trained')
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


def _partner_context(setting, trained):
    """Whether the partner chain is shown, for a trained/with/without setting."""
    return bool(trained) if setting == "trained" else setting == "with"


def _context_meta(data_args, *, partner, judge_partner):
    """Context-regime fields recorded in every generative-evaluation payload."""
    return {
        "include_partner_chain": bool(partner),
        "training_include_partner_chain": bool(data_args.include_partner_chain),
        "training_context_dropout": data_args.context_dropout,
        "eval_context_dropout": 0.0,
        "judge_include_partner_chain": bool(judge_partner),
        "include_antigen": bool(data_args.include_antigen),
    }


def run_evaluation_pipeline(method, args, supplied_config, run_config, *,
                            prepare_evaluation, evaluation_options, evaluate):
    """Load and evaluate a checkpoint using the supplied method functions.

    Steps:
      1. Resolve options and check checkpoint/config compatibility.
      2. Check the output path and record configuration/data/weight provenance.
      3. Load data, tokenizer and model; apply the generator context view.
      4. Load method state and the optional judge, then return the evaluator payload.

    Standalone errors propagate. post_training_generation handles failures
    separately to preserve previously saved training artifacts.
    """
    method = resolve(method)
    prepare_evaluation(args, run_config or supplied_config)
    argument_module = method.module('arguments')
    parser = HfArgumentParser((argument_module.ModelArguments, argument_module.DataArguments,
                              argument_module.GenerationArguments, TrainingArguments))
    model_args, data_args, _gen_args, training_args = parse_config_dict(parser, supplied_config)
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
    _, eval_dataset, tokenizer = method.module('training').get_datasets(
        model_args, data_args, training_args, backbone, checkpoint_path=args.model_dir)

    partner = _partner_context(args.partner_context, data_args.include_partner_chain)
    judge_partner = _partner_context(args.judge_partner_context, data_args.include_partner_chain)
    eval_dataset = evaluation_context_view(eval_dataset, include_partner_chain=partner)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)

    method_options = evaluation_options(args, device)

    judge_model = judge_tokenizer = judge_dataset = None
    if args.judge:
        print(f"⚖️  Loading judge model: {args.judge}")
        judge_model, judge_tokenizer, judge_dataset = build_judge(
            args.judge, args.judge_path,
            data_path=data_args.data_path,
            max_len=data_args.max_len,
            target_region=data_args.target_region,
            include_partner_chain=judge_partner,
            include_antigen=data_args.include_antigen,
            device=device,
        )

    return evaluate(
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
            **_context_meta(data_args, partner=partner, judge_partner=judge_partner),
            "compatibility": {
                "problems": problems,
                "forced": bool(problems and args.force),
            },
            "provenance": provenance,
        },
    )


def post_training_generation(method, trainer, tokenizer, eval_dataset, data_args, gen_args,
                             training_args, *, anchor_trim, method_options=None):
    """Evaluate trainer.model and return the primary payload, or None on failure.

    Steps:
      1. Load the optional judge with its requested partner-context setting.
      2. Evaluate with the training partner-context setting and dropout disabled.
      3. If requested, also evaluate with partner context removed from the generator.

    The same judge is reused across context regimes. Method-specific options are
    forwarded to the evaluator, which receives trainer.model without reloading a
    checkpoint. Evaluation errors are logged; previously saved training artifacts remain.
    """
    try:
        judge_partner = _partner_context(getattr(gen_args, 'gen_judge_partner_context', 'trained'),
                                         data_args.include_partner_chain)
        judge_model = judge_tokenizer = judge_dataset = None
        if gen_args.gen_judge:
            judge_model, judge_tokenizer, judge_dataset = build_judge(
                gen_args.gen_judge, gen_args.gen_judge_path,
                data_path=data_args.data_path, max_len=data_args.max_len,
                target_region=data_args.target_region,
                include_partner_chain=judge_partner,
                include_antigen=data_args.include_antigen, device=training_args.device)
        regimes = [(bool(data_args.include_partner_chain), artifacts.GENERATION_EVAL_FILENAME)]
        if getattr(gen_args, 'gen_eval_context_ablation', False):
            regimes.append((False, artifacts.CONTEXT_ABLATION_FILENAME))
        primary = None
        for partner, filename in regimes:
            context_dataset = evaluation_context_view(eval_dataset, include_partner_chain=partner)
            result = resolve(method).evaluate(
                trainer.model, tokenizer, context_dataset, device=training_args.device,
                num_examples=gen_args.gen_num_examples, num_samples=gen_args.gen_num_samples,
                temperature=gen_args.gen_temperature, top_k=gen_args.gen_top_k, top_p=gen_args.gen_top_p,
                stratify=gen_args.gen_stratify, seed=gen_args.gen_seed,
                skip_perplexity=gen_args.gen_skip_perplexity,
                judge_model=judge_model, judge_tokenizer=judge_tokenizer, judge_dataset=judge_dataset,
                judge_name=gen_args.gen_judge, anchor_trim=anchor_trim,
                output_path=os.path.join(training_args.output_dir, filename),
                title=f'{os.path.basename(training_args.output_dir)}   target={data_args.target_region}',
                meta={'model_dir': training_args.output_dir, 'method': method,
                      'target_region': data_args.target_region,
                      **_context_meta(data_args, partner=partner, judge_partner=judge_partner)},
                **(method_options or {}))
            if filename == artifacts.GENERATION_EVAL_FILENAME:
                primary = result
        return primary
    except Exception:
        logger.exception('Generative evaluation failed; the trained model remains saved.')
        return None
