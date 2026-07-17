"""Generative evaluation of a finished checkpoint, on the held-out eval split.

[`eval_recovery.py`](eval_recovery.py) answers "how often does the model reproduce the native loop?" (recovery).
This answers what recovery *cannot*: sample a library of candidate loops per held-out framework and
measure whether they are diverse, novel, and plausible — i.e. whether the model can **design**.

Why not test/*.json? Those are a handful of hand-assembled frameworks: far too few for a
distribution, mostly without natives, and the _lc/_ag variants hand-encode the context-injection
rules — so they drift silently the moment dataset.py changes. Here the eval set is reconstructed
from the config exactly as in training: same clustered leakage-safe split, same injection, a real
native on every example. (The curated JSONs remain a qualitative showcase via predict.py.)

train.py can run this automatically at the end of a run (config: "run_generative_eval": true) — the
shared implementation lives in cdrseqgen/eval/generation.py. This entrypoint exists because you
usually should NOT have to retrain to (re)evaluate: it scores checkpoints you already have, and lets
you re-sweep temperature/top-p or swap the judge without touching the training run.

Example:
    python eval_generative.py --config configs/h3_esm2.json \
        --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
        --num_examples 150 --stratify --num_samples 20 --top_p 0.9 \
        --judge Exscientia/IgBert
"""
import os
import argparse

import torch
from transformers import HfArgumentParser, TrainingArguments

from train import ModelArguments, DataArguments, GenerationArguments, get_datasets
from cdrseqgen.models.backbones import resolve
from cdrseqgen.eval.generation import run_generative_eval, build_judge


def main():
    ap = argparse.ArgumentParser(description="Sample + score CDR designs over the held-out eval split.")
    ap.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    ap.add_argument("--model_dir", required=True, help="Run dir holding the finetuned model.")
    ap.add_argument("--num_examples", type=int, default=200, help="Eval examples to sample (-1 = all).")
    ap.add_argument("--stratify", action="store_true", help="Balance the draw across H3 length buckets.")

    ap.add_argument("--decoding", choices=["greedy", "independent", "iterative"], default="iterative")
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
                    help="Results JSON path (default: <model_dir>/generation_eval.json).")
    args = ap.parse_args()

    # --- rebuild the exact training-time eval split from the config (same as eval_recovery.py) ---
    # GenerationArguments is in the parser only so configs carrying "run_generative_eval"/"gen_*"
    # keys still parse here; this entrypoint takes its sampling settings from the CLI instead.
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    model_args, data_args, _gen_args, training_args = parser.parse_json_file(json_file=os.path.abspath(args.config))
    training_args.output_dir = args.model_dir  # evaluate-only; nothing is written here
    training_args.report_to = []
    training_args.eval_strategy = "no"

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
        output_path=args.output or os.path.join(args.model_dir, "generation_eval.json"),
        title=f"{os.path.basename(args.model_dir)}   target={data_args.target_region}",
        meta={
            "config": args.config,
            "model_dir": args.model_dir,
            "target_region": data_args.target_region,
            "include_partner_chain": bool(data_args.include_partner_chain),
            "include_antigen": bool(data_args.include_antigen),
        },
    )


if __name__ == "__main__":
    main()
