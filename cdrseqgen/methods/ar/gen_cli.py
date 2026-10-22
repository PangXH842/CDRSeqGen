"""AR generation entrypoint called by root generate.py.

Execution:
  generate.py -> main -> common.generation.run_generation_pipeline

The shared lifecycle calls prepare_generation(), generation_state() and
generate_case() below. These supply AR validation, EOS settings and case production.
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


def main(args, config):
    """Run the generation command through the shared lifecycle and AR producers."""
    from cdrseqgen.methods.common.generation import run_generation_pipeline

    return run_generation_pipeline(
        "ar", args, config,
        prepare_generation=prepare_generation,
        generation_state=generation_state,
        generate_case=generate_case,
    )
