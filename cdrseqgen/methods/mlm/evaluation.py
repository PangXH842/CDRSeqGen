"""Configure MLM evaluation and coordinate scoring of generated CDR loops.

Entry paths:
  evaluate.py -> eval_cli.main -> common.evaluation.run_evaluation_pipeline
              -> run_generative_eval
  training -> common.evaluation.post_training_generation -> eval_cli.evaluate
  eval_cli.evaluate aliases run_generative_eval for already-loaded resources.

Functions:
  * resolve_decoding    -- choose an explicit/saved decoder or independent fallback.
  * prepare_evaluation -- validate options and set decoder/length modes in args.
  * evaluation_options -- load the optional length head and build evaluator kwargs.
  * run_generative_eval -- select examples/strategy, score and return the payload.

Main flow:
  select frameworks -> choose length strategy -> eval.generative.run_eval_loop
                    -> report -> payload + optional JSON/figure

Dependencies:
  * mlm.length: policies, fitted head and fixed-native/variable-length strategies.
  * eval.generative / eval.metrics: shared scoring, aggregation and output helpers.
  * common.evaluation: resource loading and context views.
  * mlm.recovery: separate Trainer validation metrics during training.
"""

import random

import torch

from cdrseqgen.artifacts import read_run_config
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (  # noqa: F401 (compatibility re-exports)
    SUMMARY_KEYS,
    Draw,
    aggregate,
    bucket_of,
    build_judge,
    format_report,
    log_report,
    run_eval_loop,
    score_with_judge,
    select_indices,
    write_confidence_figure,
    write_payload,
)
from cdrseqgen.eval.metrics import (
    LENGTH_BUCKETS,  # noqa: F401 (compatibility re-export)
    blosum_provenance,
)
from cdrseqgen.methods.mlm.length import (
    FixedNativeStrategy,
    LengthPredictor,
    VariableLengthStrategy,
    resolve_generation_length_policy,
)


def resolve_decoding(requested, model_dir=None, run_config=None):
    """Resolve an MLM decoder, using independent decoding as the fallback.

    Generation decoders:
      * independent -- fill all masks from one forward pass over the same prompt. (default)
      * iterative   -- fill the most-confident remaining mask per pass; filled
                       residues become context for later predictions.

    Training masking policies:
      * full_span -- hide the entire target loop. (default)
      * mixed     -- mix fully masked loops with partially revealed native loops.

    Selection:
      1. Use an explicitly requested decoder.
      2. Otherwise use a compatible saved gen_decoding.
      3. Otherwise fall back to independent, including for mixed checkpoints.

    Notes:
      * run_config takes precedence over metadata read from model_dir.
      * Missing masking metadata is treated as full_span. A recorded iterative
        choice is used automatically only with mixed training; this preserves
        the safeguard against legacy full-span iterative defaults. An explicit
        iterative request still overrides it.
      * The caller validates decoder names; this function does not reject them.
      * Prints the selection and returns the decoder name without generating
        candidates.
    """
    meta = run_config if run_config is not None else (read_run_config(model_dir) or {})
    recorded = meta.get("gen_decoding")
    policy = (meta.get("masking_policy") or "full_span").strip().lower()
    if requested is None:
        chosen = "independent"
        if policy == "mixed" and recorded in {"independent", "iterative"}:
            chosen = recorded
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
    return chosen


def prepare_evaluation(args, config):
    """Resolve MLM command options in place before the shared pipeline loads resources.

    Example: args.length_mode=None, args.decoding=None, config={}
      length_mode -> fixed_native
      decoding    -> independent

    Calls length.resolve_generation_length_policy() to check the length request,
    then resolve_decoding() to apply the explicit/saved/fallback decoder choice.
    Rejects max_new_tokens and causal decoding, which do not apply to MLM.
    Returns None; subsequent stages read the updated args.
    """
    args.length_mode = args.length_mode or 'fixed_native'
    if args.max_new_tokens is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    resolve_generation_length_policy(args.length_mode, min_length=args.min_length, max_length=args.max_length)
    args.decoding = resolve_decoding(args.decoding, run_config=config)


def evaluation_options(args, device):
    """Build run_generative_eval kwargs from the resolved command options.

    Length mode       length_predictor in returned dict
      fixed_native    None
      bounded         LengthPredictor.load(args.model_dir, device=device)
      inferred        LengthPredictor.load(args.model_dir, device=device)

    Called by the shared standalone pipeline after prepare_evaluation() and model
    loading. Also forwards decoding, bounds and length_temperature. The saved
    length head is loaded here; framework selection and generation happen later.
    """
    predictor = None
    if args.length_mode != 'fixed_native':
        predictor = LengthPredictor.load(args.model_dir, device=device)
    return dict(decoding=args.decoding, length_mode=args.length_mode, length_predictor=predictor,
                min_length=args.min_length, max_length=args.max_length,
                length_temperature=args.length_temperature)


def run_generative_eval(model, tokenizer, eval_dataset, *, device,
                        decoding="independent", num_samples=20,
                        temperature=1.0, top_k=0, top_p=1.0,
                        length_mode="fixed_native", length_predictor=None,
                        min_length=None, max_length=None,
                        length_temperature=1.0,
                        num_examples=200, stratify=False, seed=42,
                        skip_perplexity=False,
                        judge_model=None, judge_tokenizer=None,
                        judge_dataset=None, judge_name=None,
                        anchor_trim=ANCHOR_TRIM,
                        output_path=None, meta=None, title=""):
    """Generate candidate loops, score them and return an evaluation payload dict.

    Inputs:
      Loaded model/tokenizer/eval_dataset, sampling settings and device.
      Bounded/inferred modes also need a fitted length_predictor. Judge scoring
      needs judge_model, judge_tokenizer and the corresponding judge_dataset.
      eval_cli.evaluate is an alias for this function.

    Flow:
      1. model.eval(), validate anchor_trim and seed Torch/local sampling RNGs.
      2. Read target_len from the dataset and call select_indices() using
         num_examples, stratify and a seeded Python RNG.
      3. Resolve the length policy and construct one strategy for the run:
           fixed_native     -> FixedNativeStrategy
           bounded/inferred -> VariableLengthStrategy
      4. run_eval_loop() calls the strategy for each selected framework:
           draw -> sequence metrics -> optional self/judge scores
                -> positional/confidence metrics -> framework record
      5. Log the report; combine records, aggregate means, decoding/length settings,
         BLOSUM provenance and caller metadata; invoke the JSON/figure writers.

    Example: two selected frameworks, num_samples=20, both draws succeed
      records:     2 framework records, each with 20 candidate sequences
      n_generated: 2                 (framework records, not candidate count)
      summary:     metric means across contributing framework records

    Notes:
      * Native lengths support selection/reporting. Only fixed_native supplies
        the native mask count to generation; bounded/inferred use the length head.
      * skip_perplexity disables self-scoring; judge scoring is independent.
      * A strategy returning None contributes no record.
      * output_path=None skips file output; the payload is still returned.
      * Generation/scoring errors propagate. Output helpers handle some write/plot
        failures with warnings. Model evaluation mode and the Torch seed are not
        restored on return.
    """
    model.eval()
    if isinstance(anchor_trim, bool) or not isinstance(anchor_trim, int) or anchor_trim < 0:
        raise ValueError("anchor_trim must be a non-negative integer")

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # This length list selects examples; each strategy controls its generation inputs.
    lengths = [eval_dataset[i]["target_len"] for i in range(len(eval_dataset))]
    rng = random.Random(seed)
    selected = select_indices(lengths, num_examples, stratify, rng)

    length_policy = resolve_generation_length_policy(
        length_mode, min_length=min_length, max_length=max_length,
    )
    if length_policy is not None:
        if length_predictor is None:
            raise FileNotFoundError("variable-length evaluation requires a length predictor")
        strategy = VariableLengthStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, length_predictor=length_predictor,
            length_policy=length_policy, length_temperature=length_temperature,
            decoding=decoding, num_samples=num_samples, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
    else:
        strategy = FixedNativeStrategy(
            model=model, tokenizer=tokenizer, device=device, generator=generator,
            eval_dataset=eval_dataset, judge_dataset=judge_dataset,
            judge_tokenizer=judge_tokenizer, decoding=decoding, num_samples=num_samples,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )

    records = run_eval_loop(
        strategy, selected,
        model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        anchor_trim=anchor_trim,
    )

    log_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "schema_version": "cdrseqgen.generation_eval.v1",
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": decoding, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_generated": len(records),
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })

    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload
