"""Generate loops for supplied frameworks through the selected method's gen_cli.main().

To run:
  * python generate.py --model_dir <run> --pt1 EVQL...YYCAR --pt2 WGQGT... --min_length 17 --max_length 17
  * python generate.py --model_dir <run> --pt1 EVQL...YYCAR --pt2 WGQGT... --num_samples 50   # length inferred
  * python generate.py --model_dir <run> --test_file showcase/h3.json    # curated MLM cases (needs the length head)
  * python generate.py --model_dir <iglm> --method ar --model_type iglm:<iglm> --target_loop h3 --test_file cases.json

Functions:
  * build_parser -- define the generation flags; the decoder, sampling and length flags are shared
  * parse_args   -- parse argv into the generation settings
  * main         -- pick the checkpoint's method and call <method>.gen_cli.main()

Workflow:
  1. Parse the flags, then read the checkpoint's run_config.json (None for an original release).
  2. Choose the method: a saved run's declared method > --method for a release without one > MLM.
  3. <method>.gen_cli.main() -> logging
       -> common.generation.run_generation_pipeline()
       -> validate options and cases -> check compatibility -> prepare_generation -> check output path
       -> load tokenizer (warn on unencodable residues) and model -> generation_state
       -> generate_case for each case -> optional JSON; return the case records.

The decoder, sampling and length flags are shared with evaluate.py and come from
methods/common/arguments.add_sampling_arguments(); this file adds only the generation-specific ones.
Shared validation, compatibility, loading, display and output handling live in
methods/common/generation.py; each method's gen_cli.main supplies its setup, state and case
functions to that lifecycle. A case file holds one case object or a list; flanks, partner chain and
antigen are stripped and uppercased, and a malformed case or unwritable --output stops the run before
any weights load. A release without run_config.json (native IgLM) needs --method and --model_type.
This is a qualitative tool for supplied frameworks: use evaluate.py for benchmark numbers. The shared
helpers imported from common.generation stay importable here for callers of the original root module.
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
    """Return the generation argument parser.

    --model_dir is required. Cases come from --test_file or from --pt1/--pt2 and the other
    single-case flags; the decoder, sampling and length flags are the ones evaluate.py shares.
    """
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
                        help="JSON file of test cases: one case object or a list of them "
                             "(omit to use the single-case CLI args below)")

    # Fallback CLI arguments for a quick single test
    parser.add_argument("--pt1", type=str, default="",
                        help="Sequence BEFORE the missing loop (one-letter amino acids; case-insensitive)")
    parser.add_argument("--pt2", type=str, default="",
                        help="Sequence AFTER the missing loop (one-letter amino acids; case-insensitive)")
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
                        help="Where to write the results JSON ('none' to skip; default: "
                             "<model_dir>/generation_results.json). Missing directories are created.")

    # Decoder, sampling and length flags (shared with evaluate.py)
    add_sampling_arguments(parser)
    return parser


def parse_args(argv=None):
    """Parse argv (default: sys.argv[1:]) into generation settings; a bad flag exits with usage."""
    return build_parser().parse_args(argv)


# ---------- MAIN GENERATION CALLER ----------
def main(args=None):
    """Route to the checkpoint's method and return its generated case records.

    Selection order:
      checkpoint with run_config.json  -> its declared method (MLM when it declares none)
      release without run_config.json  -> --method, else MLM
      --method must agree with a saved run's method, or ValueError

    args=None parses sys.argv. The saved config is passed on without its 'source' key,
    which generation assigns itself, so a stored value cannot pose as a CLI fallback.
    Validation, loading, sampling and output all happen in the method's gen_cli.main().
    """
    # ------ 1. Parse the flags and read the checkpoint's saved metadata. ------
    args = parse_args() if args is None else args
    saved_config = read_run_config(args.model_dir)
    config = dict(saved_config or {})
    # source is assigned during generation setup; a saved field cannot claim CLI fallback.
    config.pop("source", None)

    # ------ 2. Choose the method; a saved run's declared method is authoritative. ------
    requested = getattr(args, "method", None)
    if saved_config is None and requested is not None:
        # Original releases have no CDRSeqGen run metadata. A caller must name
        # their method explicitly; declared saved methods remain authoritative.
        method = resolve(requested)
    else:
        method = checkpoint_method(config, requested)

    # ------ 3. Let the method CLI run the shared generation pipeline. ------
    return method.module('gen_cli').main(args, config)


if __name__ == "__main__":
    main()
