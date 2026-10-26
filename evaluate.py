"""Evaluate a saved checkpoint through the selected method's eval_cli.main().

Execution:
  parse_args -> read routing metadata -> eval_cli.main -> method evaluation workflow

Shared loading, compatibility, provenance and output setup live in
methods/common/evaluation.py. The method evaluator generates and scores the loops.
"""
import argparse
import json

from cdrseqgen.artifacts import read_run_config
from cdrseqgen.methods.registry import decoding_choices, resolve as resolve_method, checkpoint_method
from cdrseqgen.methods.common.logging import configure_logging
# Preserve helper imports used by callers of the original root module.
from cdrseqgen.methods.common.evaluation import (
    compatibility_problems, default_output_path, evaluation_provenance,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Sample + score CDR designs over the held-out eval split.")
    ap.add_argument("--config", required=True, help="Original config JSON used to train the run.")
    ap.add_argument("--model_dir", required=True, help="Run dir holding the finetuned model.")
    ap.add_argument("--num_examples", type=int, default=200, help="Eval examples to sample (-1 = all).")
    ap.add_argument("--stratify", action="store_true", help="Balance the draw across H3 length buckets.")

    ap.add_argument("--decoding", choices=decoding_choices(), default=None,
                    help="MLM default: a compatible saved decoder, otherwise independent. "
                         "AR/S2S use their method's decoder.")
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
    ap.add_argument("--partner_context", choices=('trained', 'with', 'without'), default='trained',
                    help="Explicit generator context ablation; evaluation always disables training context dropout.")
    ap.add_argument("--judge_partner_context", choices=('trained', 'with', 'without'), default='trained',
                    help="Judge context, independent of generator ablation; default uses the original config context.")
    ap.add_argument("--force", action="store_true",
                    help="Evaluate despite checkpoint/config input-contract mismatches. Every "
                         "mismatch is retained in the output provenance.")
    return ap.parse_args()



def main():
    configure_logging()
    args = parse_args()
    with open(args.config) as handle:
        supplied_config = json.load(handle)
    run_config = read_run_config(args.model_dir)
    method = resolve_method(supplied_config.get('method', 'mlm'))
    if run_config is not None:
        checkpoint_method(run_config, method.name)
    return method.module('eval_cli').main(args, supplied_config, run_config)


if __name__ == "__main__":
    main()
