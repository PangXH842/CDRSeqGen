"""Causal-specific CLI validation and case generation; native truth is scoring-only."""
from cdrseqgen.datasets.dataset_ar import CAUSAL_DECODE_STEPS, validate_causal_decode_steps


def _validate(args):
    if args.decoding not in (None, 'causal'):
        raise ValueError('AR checkpoints require causal decoding')
    if args.min_length is not None or args.max_length is not None:
        raise ValueError('AR currently chooses length via EOS; bounded/fixed length is unsupported')
    if args.length_temperature != 1.0:
        raise ValueError('AR has no separate length-temperature control')
    args.decoding = 'causal'


def prepare_evaluation(args, config):
    _validate(args)
    if args.length_mode not in (None, 'inferred'):
        raise ValueError('AR evaluation chooses length via EOS; use --length_mode inferred')
    args.length_mode = 'inferred'
    args.max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.get('gen_max_new_tokens', CAUSAL_DECODE_STEPS)
    validate_causal_decode_steps(args.max_new_tokens)


def evaluation_options(args, device):
    return {'max_new_tokens': args.max_new_tokens}


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
