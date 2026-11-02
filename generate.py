"""Generate loops for supplied frameworks through the selected method's gen_cli.main().

Execution:
  parse_args -> read checkpoint metadata -> gen_cli.main -> shared generation lifecycle

The decoder, sampling and length flags are shared with evaluate.py and come from
methods/common/arguments.add_sampling_arguments(); this file adds only the generation-specific ones.
Shared compatibility, loading, display and output handling live in
methods/common/generation.py. MLM's gen_cli.main supplies setup/state/case functions
from mlm.generation to that lifecycle.

Example:
  python generate.py --model_dir results/runs/example --pt1 EVQL...YYCAR --pt2 WGQGT... \
      --min_length 17 --max_length 17
"""
import argparse

from cdrseqgen.artifacts import read_run_config
from cdrseqgen.methods.common.arguments import add_sampling_arguments
from cdrseqgen.methods.registry import METHODS, checkpoint_method, resolve
# Preserve helper imports used by callers of the original root module.
from cdrseqgen.methods.common.generation import (
    LOOP_PT1_BANDS, _print_case, check_compatibility, infer_loop_number,
    load_test_cases, resolve_config,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Design CDR loops for frameworks you provide (qualitative tool; use evaluate.py for benchmarks)")
    parser.add_argument("--model_dir", "--model_path", dest="model_dir", type=str, required=True,
                        help="Path to your trained model folder (--model_path is accepted too, in both commands)")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Registered backbone name or local source; IgLM uses iglm:/local/path.")
    parser.add_argument("--method", choices=tuple(METHODS), default=None,
                        help="Method for a checkpoint without run_config.json (for native IgLM: ar). "
                             "Must agree with the method of a saved CDRSeqGen run.")

    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file of test cases (omit to use the single-case CLI args below)")

    # Fallback CLI arguments for a quick single test
    parser.add_argument("--pt1", type=str, default="", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="", help="Sequence AFTER the missing loop")
    parser.add_argument("--target_length", type=int, default=None,
                        help="Deprecated exact-length alias; prefer equal --min_length/--max_length")
    parser.add_argument("--use_native_length", action="store_true",
                        help="Explicitly use target_truth length for the fixed track; by default "
                             "target_truth is scoring-only and omitted bounds are inferred")
    parser.add_argument("--target_truth", type=str, default=None, help="Optional ground-truth loop")
    parser.add_argument("--partner_chain", type=str, default=None,
                        help="Paired partner chain (light for an h* target, heavy for l*). "
                             "Required by any checkpoint trained with include_partner_chain.")
    parser.add_argument("--antigen", type=str, default=None,
                        help="Antigen sequence; required by checkpoints trained with include_antigen. "
                             "Injection side follows the backbone's position_type automatically.")
    parser.add_argument("--target_loop", type=str, choices=['h1', 'h2', 'h3', 'l1', 'l2', 'l3'], default=None,
                        help="Which CDR this run is designing (h1/h2/h3/l1/l2/l3). A conditioned "
                             "backbone such as IgLM uses its chain identity. A mismatch with the "
                             "checkpoint's trained loop warns and is recorded in the output JSON.")
    parser.add_argument("--force", action="store_true",
                        help="Run despite blocking CONTEXT problems (the checkpoint expects partner "
                             "chain / antigen that this run does not supply, or vice versa). Loop "
                             "and shape mismatches only warn and never need this.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Length cap for the assembled input; match the training config (SAbDab2 antigen runs use 512).")
    parser.add_argument("--show_samples", type=int, default=5, help="How many sampled sequences to print per case")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the results JSON ('none' to skip; default: <model_dir>/generation_results.json)")

    # Decoder, sampling and length flags (shared with evaluate.py)
    add_sampling_arguments(parser)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main(args=None):
    args = parse_args() if args is None else args
    saved_config = read_run_config(args.model_dir)
    config = dict(saved_config or {})
    # source is assigned during generation setup; a saved field cannot claim CLI fallback.
    config.pop("source", None)
    requested = getattr(args, "method", None)
    if saved_config is None and requested is not None:
        # Original releases have no CDRSeqGen run metadata. A caller must name
        # their method explicitly; declared saved methods remain authoritative.
        method = resolve(requested)
    else:
        method = checkpoint_method(config, requested)
    return method.module('gen_cli').main(args, config)


if __name__ == "__main__":
    main()
