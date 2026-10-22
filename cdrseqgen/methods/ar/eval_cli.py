"""AR evaluation entrypoint called by root evaluate.py.

Execution:
  evaluate.py -> main -> common.evaluation.run_evaluation_pipeline

The shared lifecycle calls prepare_evaluation(), evaluation_options() and the
AR evaluate alias below. Post-training callers use evaluate with loaded resources.
"""
from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, validate_causal_decode_steps
from cdrseqgen.methods.ar.arguments import validate_inference_args as _validate
from cdrseqgen.methods.ar.evaluation import run_causal_generative_eval as evaluate


def prepare_evaluation(args, config):
    _validate(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('AR evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.get('gen_max_new_tokens', CAUSAL_DECODE_STEPS)
    validate_causal_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    return {'max_new_tokens': args.max_new_tokens}


def main(args, config, run_config=None):
    """Run standalone evaluation through the shared lifecycle and AR evaluator."""
    from cdrseqgen.methods.common.evaluation import run_evaluation_pipeline

    return run_evaluation_pipeline(
        "ar", args, config, run_config,
        prepare_evaluation=prepare_evaluation,
        evaluation_options=evaluation_options,
        evaluate=evaluate,
    )
