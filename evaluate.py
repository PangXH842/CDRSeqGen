"""Evaluate a saved checkpoint through the selected method's eval_cli.main().

Execution:
  parse_args -> read routing metadata -> eval_cli.main -> method evaluation workflow

The decoder, sampling and length flags are shared with generate.py and come from
methods/common/arguments.add_sampling_arguments(); this file adds only the evaluation-specific ones.
Shared loading, compatibility, provenance and output setup live in methods/common/evaluation.py.
The method evaluator generates and scores the loops.
"""
import argparse
import json

from cdrseqgen.artifacts import read_run_config
from cdrseqgen.methods.common.arguments import PARTNER_CONTEXTS, add_sampling_arguments
from cdrseqgen.methods.registry import DEFAULT_METHOD, checkpoint_method, resolve
# Re-export shared helpers for callers importing them from this root module.
from cdrseqgen.methods.common.evaluation import (
    compatibility_problems, default_output_path, get_evaluation_source_info,
)


def build_parser():
    parser = argparse.ArgumentParser(description="Sample + score CDR designs over the held-out eval split.")
    parser.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    parser.add_argument("--model_dir", "--model_path", dest="model_dir", required=True,
                        help="Run dir holding the finetuned model (--model_path is accepted too, in both commands).")
    parser.add_argument("--num_examples", type=int, default=150, help="Eval examples to sample (-1 = all).")
    parser.add_argument("--stratify", action="store_true", help="Balance the draw across H3 length buckets.")
    parser.add_argument("--length_mode", choices=("fixed_native", "bounded", "inferred"), default=None,
                        help="MLM length track: fixed_native (default; the native mask count), bounded "
                             "(needs --min_length and/or --max_length) or inferred (the length head "
                             "chooses). AR/S2S always infer length via <eos>.")
    add_sampling_arguments(parser)

    parser.add_argument("--judge", type=str, default=None,
                        help="model_type of an INDEPENDENT LM to score every candidate "
                             "(makes perplexity comparable ACROSS models; self-ppl is not).")
    parser.add_argument("--judge_path", type=str, default=None, help="Weights for the judge (default: the --judge hub id).")
    parser.add_argument("--output", type=str, default=None,
                        help="Results JSON path ('none' to skip writing; default: a parameter-addressed "
                             "file under <model_dir>/evaluations/).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output with the same path/parameters.")
    parser.add_argument("--partner_context", choices=PARTNER_CONTEXTS, default='trained',
                        help="Explicit generator context ablation; evaluation always disables training context dropout.")
    parser.add_argument("--judge_partner_context", choices=PARTNER_CONTEXTS, default='trained',
                        help="Judge context, independent of generator ablation; default uses the original config context.")
    parser.add_argument("--force", action="store_true",
                        help="Evaluate despite checkpoint/config input-contract mismatches. Every "
                             "mismatch is retained in the output provenance.")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    with open(args.config) as handle:
        supplied_config = json.load(handle)
    run_config = read_run_config(args.model_dir)
    method = resolve(supplied_config.get('method', DEFAULT_METHOD))
    if run_config is not None:
        checkpoint_method(run_config, method.name)
    return method.module('eval_cli').main(args, supplied_config, run_config)


if __name__ == "__main__":
    main()
