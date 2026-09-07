"""Masked-specific options for the shared generation/evaluation commands."""
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy
from cdrseqgen.methods.mlm.length_predictor import LengthPredictor


def prepare_evaluation(args, config):
    from evaluate import resolve_decoding
    args.length_mode = args.length_mode or 'fixed_native'
    if args.max_new_tokens is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    resolve_generation_length_policy(args.length_mode, min_length=args.min_length, max_length=args.max_length)
    args.decoding = resolve_decoding(args.decoding, run_config=config)


def evaluation_options(args, device):
    predictor = None
    if args.length_mode != 'fixed_native':
        predictor = LengthPredictor.load(args.model_dir, device=device)
    return dict(decoding=args.decoding, length_mode=args.length_mode, length_predictor=predictor,
                min_length=args.min_length, max_length=args.max_length,
                length_temperature=args.length_temperature)


def prepare_generation(args, cases, config):
    from generate import resolve_case_length
    if getattr(args, 'max_new_tokens', None) is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    args.decoding = args.decoding or 'iterative'  # preserve the qualitative tool's existing default
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    return [resolve_case_length(case, args)[1] for case in cases]


def generation_state(args, policies, device):
    return LengthPredictor.load(args.model_path, device=device) if any(p.requires_predictor for p in policies) else None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    from generate import generate_case as masked_case
    return masked_case(model, tokenizer, device, args, generator, case, state)
