"""Generate loops for supplied frameworks through the selected method's gen_cli.main().

Execution:
  parse arguments -> read checkpoint metadata -> gen_cli.main -> shared generation lifecycle

Shared compatibility, loading, display and output handling live in
methods/common/generation.py. MLM's gen_cli.main supplies setup/state/case functions
from mlm.generation to that lifecycle.

Example:
  python generate.py --model_path results/runs/example --pt1 EVQL...YYCAR --pt2 WGQGT... \
      --min_length 17 --max_length 17
"""
import argparse

from cdrseqgen.methods.common.logging import configure_logging
from cdrseqgen.methods.registry import decoding_choices, checkpoint_method
# Preserve helper imports used by callers of the original root module.
from cdrseqgen.methods.common.generation import (
    LOOP_PT1_BANDS, check_compatibility, get_configs, get_test_case,
    infer_loop_number, _print_case,
)


def main(args):
    configure_logging()
    print(f"🧬 Checkpoint: {args.model_path}")
    configs = get_configs(args)
    method = checkpoint_method(configs)
    return method.module('gen_cli').main(args, configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Design CDR loops for frameworks you provide (qualitative tool; use evaluate.py for benchmarks)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your trained model folder")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Backbone: 'scratch', an ESM-2 id, a protein-BERT id, or any HF masked-LM hub path")

    parser.add_argument("--test_file", type=str, default=None,
                        help="JSON file of test cases (omit to use the single-case CLI args below)")

    # Fallback CLI arguments for a quick single test
    parser.add_argument("--pt1", type=str, default="", help="Sequence BEFORE the missing loop")
    parser.add_argument("--pt2", type=str, default="", help="Sequence AFTER the missing loop")
    parser.add_argument("--min_length", type=int, default=None,
                        help="Inclusive minimum CDR length (alone or with --max_length)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Inclusive maximum CDR length (alone or with --min_length)")
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
                        help="Which CDR this run is designing (h1/h2/h3/l1/l2/l3). Declarative: it "
                             "cannot change the computation, and a mismatch with the checkpoint's "
                             "trained loop warns rather than refuses (off-target probing is a valid "
                             "experiment). The mismatch is recorded in the output JSON.")
    parser.add_argument("--force", action="store_true",
                        help="Run despite blocking CONTEXT problems (the checkpoint expects partner "
                             "chain / antigen that this run does not supply, or vice versa). Loop "
                             "and shape mismatches only warn and never need this.")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Length cap for the assembled input; match the training config (SAbDab2 antigen runs use 512).")

    # Sampling controls
    parser.add_argument("--decoding", choices=decoding_choices(), default=None,
                        help="Decoder for the selected method. MLM: independent (one pass) or "
                             "iterative (one residue per pass); default is a compatible saved "
                             "choice, otherwise independent.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="AR decode cap, including EOS.")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Candidate loops to sample per framework (collapses to 1 when --temperature 0)")
    parser.add_argument("--show_samples", type=int, default=5, help="How many sampled sequences to print per case")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (>1 flatter, <1 sharper)")
    parser.add_argument("--length_temperature", type=float, default=1.0,
                        help="Temperature for sampled length classes; 0 selects the most likely "
                             "allowed length")
    parser.add_argument("--top_k", type=int, default=0, help="Keep only the top-k logits (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling cumulative-prob cutoff (1.0 = disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling")
    parser.add_argument("--skip_perplexity", action="store_true",
                        help="Skip pseudo-perplexity scoring (the compute-heavy per-candidate part)")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to write the results JSON ('none' to skip; default: <model_path>/generation_results.json)")

    main(parser.parse_args())
