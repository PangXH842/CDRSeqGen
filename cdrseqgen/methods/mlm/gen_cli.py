"""MLM command functions called by root generate.py for supplied frameworks.

Caller:
  generate.py -> methods.registry -> method.module('gen_cli')

Command path:
  prepare_generation -> generation_state -> generate_case (per case)

Functions:
  * prepare_generation -- validate options, set the decoder default and return
                          one length policy per case
  * generation_state   -- load the length head if any case requires it, else None
  * generate_case      -- generate/score one case through case_generation;
                          return a JSON-serialisable record
  * resolve_case_length -- helper for prepare_generation and generate_case:
                           return normalized native truth and its length policy

Notes:
  * Root generate.py parses the CLI, loads the model/tokenizer, coordinates
    the cases and writes the output JSON.
  * Case-level length settings override command settings. Native truth is used
    for scoring; its length enters generation only with use_native_length.
  * Generation currently defaults to iterative decoding.
"""
from cdrseqgen.methods.mlm.length import resolve_length_policy
from cdrseqgen.methods.mlm.length_predictor import LengthPredictor


def resolve_case_length(case, args):
    """Return normalized native sequence and its independent generation policy.

    Native truth is retained for scoring but reveals its length only when
    ``use_native_length`` is explicitly requested.
    """
    native = case.get("target_truth")
    native = native.upper().strip() if native else None

    def case_value(key):
        return case[key] if key in case else getattr(args, key, None)

    policy = resolve_length_policy(
        native_length=len(native) if native else None,
        use_native_length=bool(case_value("use_native_length")),
        target_length=case_value("target_length"),
        min_length=case_value("min_length"),
        max_length=case_value("max_length"),
    )
    return native, policy


def prepare_generation(args, cases, config):
    if getattr(args, 'max_new_tokens', None) is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    args.decoding = args.decoding or 'iterative'  # preserve the qualitative tool's existing default
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    return [resolve_case_length(case, args)[1] for case in cases]


def generation_state(args, policies, device):
    return LengthPredictor.load(args.model_path, device=device) if any(p.requires_predictor for p in policies) else None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Generate + score one test case. Returns a JSON-serialisable record."""
    from cdrseqgen.methods.mlm.case_generation import generate_case_record

    native, length_policy = resolve_case_length(case, args)
    return generate_case_record(
        model,
        tokenizer,
        device,
        args,
        generator,
        case,
        native=native,
        length_policy=length_policy,
        length_predictor=state,
        print_callback=print_callback,
    )
