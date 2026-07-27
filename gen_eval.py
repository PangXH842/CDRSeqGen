"""Generative evaluation of a finished checkpoint, on the held-out eval split. 

Entrypoint used to evaluate trained model checkpoints, especially CDR generation capabilities. 
Recommended to set up judge to measure perplexity since its not automatically done in train.py. 

Usage example:
    python eval_generative.py --config configs/h3_esm2.json \
        --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
        --num_examples 150 --stratify --num_samples 20 --top_p 0.9 \
        --judge Exscientia/IgBert
"""
import os
import json
import argparse

import torch
from transformers import HfArgumentParser, TrainingArguments

from train import ModelArguments, DataArguments, GenerationArguments, get_datasets, RUN_META_FILENAME
from cdrseqgen.models.registry import resolve
from cdrseqgen.eval.generation import run_generative_eval, build_judge


def read_run_meta(model_dir):
    """Load the run_meta.json a training run wrote, or None (older runs won't have it)."""
    path = os.path.join(model_dir, RUN_META_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def check_target_region(model_dir, config_target):
    """Report the CDR loop the checkpoint was trained for, and warn if the config disagrees.

    The eval runs on config_target (the dataset is built from --config). If the checkpoint's own
    run_meta says it was trained for a DIFFERENT loop, we're masking the wrong CDR — the numbers
    would be confident and meaningless — so warn loudly. Returns the model's trained loop (or None).
    """
    meta = read_run_meta(model_dir)
    trained_for = meta.get("target_region") if meta else None
    if trained_for is None:
        print(f"ℹ️  No {RUN_META_FILENAME} in {model_dir} (older run) — can't verify the model's trained loop.")
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

    ap.add_argument("--decoding", choices=["independent", "iterative"], default="iterative")
    ap.add_argument("--num_samples", type=int, default=20, help="Candidates per framework.")
    ap.add_argument("--temperature", type=float, default=1.0)
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

    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, _gen_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config))
    training_args.output_dir = args.model_dir  
    training_args.report_to = []
    training_args.eval_strategy = "no"

    # Provenance: what CDR was this checkpoint actually trained for? (warns on config mismatch)
    trained_for = check_target_region(args.model_dir, data_args.target_region)

    backbone = resolve(model_args.model_type)
    _, eval_dataset, tokenizer = get_datasets(model_args, data_args, training_args, backbone)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = backbone.load_pretrained_model(args.model_dir)
    model.to(device)

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
            "model_trained_for": trained_for,                # loop the checkpoint was trained for (run_meta)
            "include_partner_chain": bool(data_args.include_partner_chain),
            "include_antigen": bool(data_args.include_antigen),
        },
    )


if __name__ == "__main__":
    main()
