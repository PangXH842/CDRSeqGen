"""Generate MLM CDR candidates and case records for supplied frameworks.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Functions:
  * prepare_generation -- validate options, select a decoder and resolve case lengths
  * generation_state   -- load one saved length head if any case needs it
  * generate_case      -- sample one case, calculate measurements and return its record

Notes:
  * gen_cli owns logging; common.generation loads cases/model/tokenizer, seeds
    sampling and writes the output. This module supplies the three method hooks.
  * length.py owns policies, prompts, length selection and span sampling;
    sampling.py supplies independent/iterative decoding and self-scoring.
  * Native truth is optional. It supports reference scores; its length affects
    generation only when use_native_length is explicitly enabled.
  * These are supplied-case results. Dataset selection, judge scoring and
    aggregation across evaluation frameworks belong to the evaluation workflow.
"""

from cdrseqgen.eval.metrics import amino_acid_recovery, calculate
from cdrseqgen.eval.generative import normalize_scores
from cdrseqgen.methods.mlm.arguments import validate_inference_args
from cdrseqgen.methods.mlm.evaluation import resolve_decoding
from cdrseqgen.methods.mlm.length import (
    LengthPredictor, resolve_case_length, sample_length_conditioned_spans,
)
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood


def prepare_generation(args, cases, config):
    """Resolve MLM command settings before the shared pipeline loads model weights.

    Steps:
      1. validate_inference_args(): reject max_new_tokens and decoder names MLM lacks.
      2. resolve_decoding(): set args.decoding from an explicit request or a
         compatible saved setting; otherwise independent.
      3. resolve_case_length() -> one LengthPolicy per case, in input order.

    Example: cases request exact length 8 and inferred length
      return: [fixed policy, inferred policy]
      later:  generation_state() sees that a length head is required

    Recorded iterative decoding is selected automatically only for mixed masking.
    This function updates args.decoding and checks length settings; it does not
    load a model/head or generate candidates.
    """
    validate_inference_args(args)
    args.decoding = resolve_decoding(args.decoding, run_config=config)
    return [resolve_case_length(case, args)[1] for case in cases]


def generation_state(args, policies, device):
    """Return one saved length head for all cases that need it, or None.

    Policies                    Returned state
      fixed only                None
      any bounded or inferred   LengthPredictor.load(args.model_dir, device=device)

    The shared pipeline calls this after loading the sequence model and seeding
    sampling. The resulting state is passed to every generate_case() call.
    """
    return (
        LengthPredictor.load(args.model_dir, device=device)
        if any(p.requires_predictor for p in policies) else None
    )


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Return candidates, measurements and length metadata for one supplied case.

    Steps:
      1. Resolve native truth, length policy, flanks and optional context.
      2. sample_length_conditioned_spans() -> prompts, lengths and residue samples.
      3. Decode candidates, calculate sequence metrics and optionally self-score.
      4. Call print_callback() and return the JSON-ready case record.

    state is the optional saved length head from generation_state(). Native truth
    is optional; per-candidate AAR requires a native-length match, and a candidate of
    another length records recovery=None, as in the AR/S2S case records.
    decoding.num_samples counts the retained candidates. skip_perplexity disables
    generator pseudo-perplexity. No external judge is loaded here.
    """
    # ------ 1. Resolve this case's length policy and conditioning inputs. ------
    native, length_policy = resolve_case_length(case, args)
    name = case.get("name", "Unnamed Test")
    pt1, pt2 = case["pt1"], case["pt2"]
    partner = case.get("partner_chain") or args.partner_chain
    antigen = case.get("antigen") or args.antigen

    # ------ 2. Choose lengths, assemble prompts and sample the masked spans. ------
    generated, length_metadata = sample_length_conditioned_spans(
        model,
        tokenizer,
        device,
        generator,
        pt1=pt1,
        pt2=pt2,
        policy=length_policy,
        length_predictor=state,
        partner_chain=partner,
        antigen=antigen,
        max_len=args.max_len,
        decoding=args.decoding,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        length_temperature=getattr(args, "length_temperature", 1.0),
    )

    # ------ 3. Measure candidates, using native truth only where available. ------
    samples = []
    for generated_span in generated:
        seq = tokenizer.decode(generated_span.token_ids).upper().strip()
        sample = {"seq": seq, "length": len(seq)}
        if native:
            sample["length_match"] = len(seq) == len(native)
            sample["recovery"] = amino_acid_recovery(seq, native) if sample["length_match"] else None
        if not args.skip_perplexity:
            _, perplexity = pseudo_log_likelihood(
                model,
                generated_span.input_ids,
                generated_span.mask_positions,
                generated_span.token_ids,
                tokenizer.special_ids,
                tokenizer.mask_id,
                device,
            )
            sample["pseudo_perplexity"] = perplexity
        samples.append(sample)

    if not args.skip_perplexity:
        scores = normalize_scores([sample["pseudo_perplexity"] for sample in samples],
                                  len(samples), label="MLM case self scorer")
        for sample, score in zip(samples, scores):
            if score is None:
                sample.pop("pseudo_perplexity")
            else:
                sample["pseudo_perplexity"] = score
    seqs = [sample["seq"] for sample in samples]
    metrics = calculate(seqs, native)
    perplexities = [sample["pseudo_perplexity"] for sample in samples if "pseudo_perplexity" in sample]
    if perplexities:
        metrics["pseudo_perplexity_mean"] = sum(perplexities) / len(perplexities)
        metrics["pseudo_perplexity_min"] = min(perplexities)
        metrics["pseudo_perplexity_max"] = max(perplexities)

    # ------ 4. Display the case and return its candidates and metadata. ------
    generated_lengths = [sample["length"] for sample in samples]
    distinct_lengths = sorted(set(generated_lengths))
    gap_display = "/".join(map(str, distinct_lengths)) or "—"
    print_callback(name, pt1, pt2, gap_display, native, args, samples, metrics)

    return {
        "name": name,
        "pt1": pt1,
        "pt2": pt2,
        "target_length": distinct_lengths[0] if len(distinct_lengths) == 1 else None,
        "generated_lengths": generated_lengths,
        "length_policy": length_metadata,
        "native": native,
        "decoding": {
            "mode": args.decoding,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "num_samples": len(samples),
            "seed": args.seed,
        },
        "samples": samples,
        "metrics": metrics,
    }
