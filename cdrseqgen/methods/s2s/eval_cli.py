"""S2S command functions called by root evaluate.py for a saved checkpoint.

Caller:
  evaluate.py -> methods.registry -> method.module('eval_cli')

Command path:
  prepare_evaluation -> evaluation_options -> evaluate

Functions:
  * prepare_evaluation -- validate decoder/length options and resolve the EOS cap
  * evaluation_options -- return the evaluator's max_new_tokens setting
  * evaluate           -- run s2s.evaluation.run_s2s_generative_eval

Notes:
  * Shared command checks live in arguments.validate_inference_args.
  * Loop length is chosen by EOS; native truth is retained for scoring.
  * Root evaluate.py owns parsing, model/data/judge loading and output setup.
"""
from cdrseqgen.datasets.dataset_s2s import S2S_DECODE_STEPS, validate_s2s_decode_steps
from cdrseqgen.methods.s2s.arguments import validate_inference_args as _validate
from cdrseqgen.methods.s2s.evaluation import run_s2s_generative_eval as evaluate


def prepare_evaluation(args, config):
    _validate(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('s2s evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.get('gen_max_new_tokens', S2S_DECODE_STEPS)
    validate_s2s_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    return {'max_new_tokens': args.max_new_tokens}
