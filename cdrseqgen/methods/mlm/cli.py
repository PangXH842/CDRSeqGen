"""Masked-specific options for the shared generation/evaluation commands.

Everything here is a masking assumption that ``generate.py``/``evaluate.py`` must not carry: the
independent/iterative decoder choice, the explicit length policies, and the gap-sentinel length
head. The causal side answers the same four questions in ``methods/ar/cli.py`` and shares nothing
but the protocol — which is the point. The shared entrypoints call this module through
``registry.resolve(...).module('cli')`` and import nothing from a method package themselves.
"""
from cdrseqgen.artifacts import read_run_config
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy, resolve_length_policy
from cdrseqgen.methods.mlm.length_predictor import LengthPredictor


def resolve_decoding(requested, model_dir=None, run_config=None):
    """Pick a decoder that is compatible with the checkpoint's masking policy.

    Missing policy metadata means an older full-span checkpoint. A recorded iterative decoder must
    not silently override that training contract; callers can still request it explicitly for an
    intentional off-distribution experiment.
    """
    meta = run_config if run_config is not None else (read_run_config(model_dir) or {})
    recorded = meta.get("gen_decoding")
    policy = (meta.get("masking_policy") or "full_span").strip().lower()
    if requested is None:
        if policy == "mixed":
            chosen = recorded if recorded in {"independent", "iterative"} else "iterative"
        else:
            chosen = "independent"
        source = f"masking_policy={policy!r}"
        if recorded and recorded != chosen:
            source += f"; ignored incompatible recorded gen_decoding={recorded!r}"
        print(f"🎲 Decoding: {chosen} ({source})")
    else:
        chosen = requested
        if recorded and recorded != chosen:
            print(f"🎲 Decoding: {chosen} (CLI override; the run recorded '{recorded}')")
        else:
            print(f"🎲 Decoding: {chosen}")
    if chosen == "iterative" and policy != "mixed":
        print(f"⚠️  Iterative decoding on a {policy} checkpoint: the model never saw a partially "
              "revealed loop in training, so this is off distribution and roughly halves core "
              "recovery. Train with masking_policy: mixed for iterative decoding.")
    return chosen


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


def prepare_evaluation(args, config):
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
