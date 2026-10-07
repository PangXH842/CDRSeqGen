"""AR command functions called by root generate.py for supplied frameworks.

Caller:
  generate.py -> methods.registry -> method.module('gen_cli')

Command path:
  prepare_generation -> generation_state -> generate_case (per case)

Functions:
  * prepare_generation -- validate options/cases and resolve the EOS decode cap
  * generation_state   -- return None; this method has no separate length head
  * generate_case      -- generate/score one case through case_generation

Notes:
  * Shared command checks live in arguments.validate_inference_args.
  * Loop length is chosen by EOS; native truth is retained for scoring.
  * Root generate.py owns parsing, model loading and output writing.
"""
from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, validate_causal_decode_steps
from cdrseqgen.methods.ar.arguments import validate_inference_args as _validate


def prepare_generation(args, cases, config):
    _validate(args)
    for case in cases:
        for key in ('min_length', 'max_length', 'target_length', 'use_native_length'):
            value = case.get(key, getattr(args, key, None))
            if value is not None and value is not False:
                raise ValueError(f'AR chooses length via EOS; {key} is unsupported')
    args.max_new_tokens = getattr(args, 'max_new_tokens', None)
    if args.max_new_tokens is None:
        args.max_new_tokens = config.get('gen_max_new_tokens', CAUSAL_DECODE_STEPS)
    validate_causal_decode_steps(args.max_new_tokens)
    return None


def generation_state(args, policies, device):
    return None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    from cdrseqgen.methods.ar.case_generation import generate_case_record
    return generate_case_record(model, tokenizer, device, args, generator, case, print_callback)
