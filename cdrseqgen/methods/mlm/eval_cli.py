"""MLM command functions called by root evaluate.py for a saved checkpoint.

Caller:
  evaluate.py -> methods.registry -> method.module('eval_cli')

Command path:
  prepare_evaluation -> evaluation_options -> evaluate

Functions:
  * prepare_evaluation -- validate/resolve evaluation options in args
  * evaluation_options -- return evaluator kwargs; load the length head when
                          the requested length mode requires it
  * evaluate           -- run mlm.evaluation.run_generative_eval
  * resolve_decoding   -- helper for prepare_evaluation: choose a decoder from
                          checkpoint masking metadata or an explicit override

Notes:
  * Root evaluate.py parses the CLI, loads the model/tokenizer and eval dataset,
    prepares the judge and chooses the output path.
  * The default decoder follows the checkpoint masking policy. Variable-length
    evaluation loads the saved length head; fixed_native supplies native length.
"""
from cdrseqgen.artifacts import read_run_config
from cdrseqgen.methods.mlm.evaluation import run_generative_eval as evaluate
from cdrseqgen.methods.mlm.length import resolve_generation_length_policy
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
