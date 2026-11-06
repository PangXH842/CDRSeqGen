"""Evaluate generated MLM CDR candidates against a selected reference dataset.

Entry paths:
  evaluate.py -> eval_cli.main -> common.evaluation.run_evaluation_pipeline
              -> run_generative_eval
  training.finish_training -> common.evaluation.post_training_generation
                           -> eval_cli.evaluate (alias of run_generative_eval)

Functions:
  * prepare_evaluation  -- validate options and resolve decoder/length modes
  * evaluation_options  -- load an optional length head and return evaluator kwargs
  * resolve_decoding    -- choose an explicit/saved decoder or independent fallback
  * run_generative_eval -- select frameworks/strategy, score and return a payload

Notes:
  * common.evaluation loads standalone resources and controls context regimes;
    this module receives a loaded model, tokenizer, dataset and optional judge.
  * mlm.length owns policies, the fitted head and both evaluation strategies;
    mlm.sampling supplies residue decoding and generator pseudo-perplexity.
  * fixed_native exposes the native mask count; bounded/inferred use the length
    head. Native references also support framework selection and scoring.
  * eval.generative / eval.metrics own shared scoring, aggregation and reports.
    Trainer validation recovery lives separately in mlm.recovery.
"""

import random

import torch

from cdrseqgen.artifacts import GENERATION_EVAL_SCHEMA, read_run_config
from cdrseqgen.eval import ANCHOR_TRIM
from cdrseqgen.eval.generative import (
    SUMMARY_KEYS,
    aggregate,
    log_report,
    run_eval_loop,
    select_indices,
    write_confidence_figure,
    write_payload,
)
from cdrseqgen.eval.metrics import blosum_provenance
from cdrseqgen.methods.common.arguments import validate_anchor_trim
from cdrseqgen.methods.mlm.arguments import validate_inference_args
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
      * Decoder names are validated by arguments.validate_inference_args(), not here.
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

    Calls arguments.validate_inference_args() to reject AR-only options and non-MLM decoders, then
    length.resolve_generation_length_policy() to check the length request, then
    resolve_decoding() to apply the explicit/saved/fallback decoder choice.
    Returns None; subsequent stages read the updated args.
    """
    args.length_mode = args.length_mode or 'fixed_native'
    validate_inference_args(args)
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
                        num_examples=150, stratify=False, seed=42,
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

    Steps:
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
      n_selected:  2                 (frameworks chosen from the eval dataset)
      n_generated: 2                 (framework records, not candidate count)
      summary:     metric means across contributing framework records

    The payload carries the same method, n_selected and schema_version keys as the AR/S2S
    payload. It has no generation_validity block: every draw is retained, because the
    length is chosen before decoding rather than discovered by an end-of-sequence token.

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
    # ------ 1. Prepare the loaded model, validate trim and seed sampling. ------
    model.eval()
    validate_anchor_trim(anchor_trim)

    torch.manual_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # ------ 2. Select frameworks, optionally stratifying by native length. ------
    lengths = [eval_dataset[i]["target_len"] for i in range(len(eval_dataset))]
    rng = random.Random(seed)
    selected = select_indices(lengths, num_examples, stratify, rng)

    # ------ 3. Resolve the length policy and construct the candidate strategy. ------
    shared = dict(
        model=model, tokenizer=tokenizer, device=device, generator=generator,
        eval_dataset=eval_dataset, judge_dataset=judge_dataset, judge_tokenizer=judge_tokenizer,
        decoding=decoding, num_samples=num_samples,
        temperature=temperature, top_k=top_k, top_p=top_p,
    )
    length_policy = resolve_generation_length_policy(
        length_mode, min_length=min_length, max_length=max_length,
    )
    if length_policy is not None:
        if length_predictor is None:
            raise FileNotFoundError("variable-length evaluation requires a length predictor")
        strategy = VariableLengthStrategy(
            length_predictor=length_predictor, length_policy=length_policy,
            length_temperature=length_temperature, **shared,
        )
    else:
        strategy = FixedNativeStrategy(**shared)

    # ------ 4. Generate and score candidates for the selected frameworks. ------
    records = run_eval_loop(
        strategy, selected,
        model=model, tokenizer=tokenizer, device=device,
        skip_perplexity=skip_perplexity,
        judge_model=judge_model, judge_tokenizer=judge_tokenizer,
        anchor_trim=anchor_trim,
    )

    # ------ 5. Report, aggregate and optionally write the evaluation payload. ------
    log_report(records, title=title, judge_name=judge_name)

    payload = dict(meta or {})
    payload.update({
        "schema_version": GENERATION_EVAL_SCHEMA,
        # A BLOSUM number without its matrix and gap penalties is not reproducible.
        "blosum": blosum_provenance(),
        "method": "mlm",
        "judge": judge_name,
        "length_policy": strategy.payload_length_policy(),
        "anchor_trim_each_end": anchor_trim,
        "decoding": {
            "mode": decoding, "num_samples": num_samples, "temperature": temperature,
            "top_k": top_k, "top_p": top_p, "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_selected": len(selected),
        "n_generated": len(records),
        "summary": {key: aggregate(records, key)[0] for key, _ in SUMMARY_KEYS},
        "records": records,
    })

    write_payload(payload, output_path, len(records))
    write_confidence_figure(payload, output_path, title)
    return payload
