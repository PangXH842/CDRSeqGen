"""Evaluate a saved checkpoint through the selected method's eval_cli.main().

To run:
  * python evaluate.py --config configs/mlm/heavy/h3_scratch.json --model_dir <run>   # the config it trained with
  * python evaluate.py --config configs/mlm/heavy/h3_scratch.json --model_dir <run> --num_examples -1 --stratify
  * python evaluate.py --config configs/mlm/heavy/h3_scratch.json --model_dir <run> --judge Exscientia/IgBert
  * python evaluate.py --config configs/mlm/heavy/h3_scratch.json --model_dir <run> --length_mode inferred
  * python evaluate.py --config configs/mlm/heavy/h3_scratch.json --model_dir <run> --output none   # report only

Functions:
  * build_parser -- define the evaluation flags; the decoder, sampling and length flags are shared
  * parse_args   -- parse argv into the evaluation settings
  * main         -- pick the checkpoint's method and call <method>.eval_cli.main()

Workflow:
  1. Parse the flags, then read the --config JSON and the checkpoint's run_config.json.
  2. Choose the method: the config's method > MLM. A saved run_config.json must name the same method.
  3. <method>.eval_cli.main() -> logging
       -> common.evaluation.run_evaluation_pipeline()
       -> validate options -> prepare_evaluation -> check compatibility and output path
       -> load data and model -> evaluation_options -> load judge (optional)
       -> run_generative_eval -> candidate records, summary and optional JSON/figure output.

The decoder, sampling and length flags are shared with generate.py and come from
methods/common/arguments.add_sampling_arguments(); this file adds only the evaluation-specific ones.
Shared validation, loading, compatibility, provenance and output setup live in
methods/common/evaluation.py, whose checks run before any model loads; the dataset split named in
--config decides validation versus test use. The method evaluator generates and scores the loops,
and MLM, AR and S2S write one payload schema. The shared helpers imported from common.evaluation
stay importable here for callers of the original root module.
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
    """Return the evaluation argument parser.

    --config and --model_dir are required. The decoder, sampling and length flags are the ones
    generate.py shares; the others pick the frameworks (--num_examples, --stratify), the MLM
    length track, the independent judge, the output path and the context/compatibility overrides.
    """
    parser = argparse.ArgumentParser(description="Generate and score CDR designs on the configured validation/test split.")
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
    """Parse argv (default: sys.argv[1:]) into evaluation settings; a bad flag exits with usage."""
    return build_parser().parse_args(argv)


# ---------- MAIN EVALUATION CALLER ----------
def main(args=None):
    """Route to the checkpoint's method and return its evaluation payload.

    Selection order:
      --config's method field          -> MLM when it declares none
      a saved run_config.json must declare that same method, or ValueError

    args=None parses sys.argv. The config and run_config.json are read here only to route;
    the method CLI receives both again for parsing, compatibility checks and provenance.
    """
    # ------ 1. Parse the flags and read the routing metadata. ------
    args = parse_args() if args is None else args
    with open(args.config) as handle:
        supplied_config = json.load(handle)
    if not isinstance(supplied_config, dict):
        raise ValueError("--config must contain a JSON object")
    run_config = read_run_config(args.model_dir)

    # ------ 2. Choose the method; a saved run must agree with it. ------
    method = resolve(supplied_config.get('method', DEFAULT_METHOD))
    if run_config is not None:
        checkpoint_method(run_config, method.name)

    # ------ 3. Let the method CLI run the shared evaluation pipeline. ------
    return method.module('eval_cli').main(args, supplied_config, run_config)


if __name__ == "__main__":
    main()
