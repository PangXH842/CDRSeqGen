"""Configure MLM generation and build scored records for supplied frameworks.

Entry path:
  generate.py -> gen_cli.main -> common.generation.run_generation_pipeline
              -> prepare_generation -> generation_state -> generate_case

Callbacks:
  * prepare_generation -- validate decoder options and resolve each case's policy.
  * generation_state   -- load one saved length head if any policy needs it.
  * generate_case      -- resolve native truth/policy and build one scored record.

Candidate flow:
  generate_case -> length.sample_length_conditioned_spans -> sampling decoders
                -> metrics and optional self-scoring -> display callback -> record

length.py owns policies, prompt construction, length selection and span sampling.
common.generation loads cases/model/tokenizer, seeds sampling and writes the outer
payload. gen_cli.main configures logging; this module installs no handlers.
"""

from cdrseqgen.eval.metrics import amino_acid_recovery, calculate
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
    """Generate, score and serialize one fixed, bounded or inferred-length case.

    state is the optional length head returned by generation_state(). Native truth
    is retained for scoring; an explicit native-length policy may also use its length.
    print_callback receives the case summary before the JSON-ready record is returned.
    """
    native, length_policy = resolve_case_length(case, args)
    name = case.get("name", "Unnamed Test")
    pt1, pt2 = case["pt1"], case["pt2"]
    partner = case.get("partner_chain") or args.partner_chain
    antigen = case.get("antigen") or args.antigen

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

    samples = []
    for generated_span in generated:
        seq = tokenizer.decode(generated_span.token_ids).upper().strip()
        entry = {"seq": seq, "length": len(seq)}
        if native:
            entry["length_match"] = len(seq) == len(native)
            if entry["length_match"]:
                entry["recovery"] = amino_acid_recovery(seq, native)
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
            entry["pseudo_perplexity"] = perplexity
        samples.append(entry)

    seqs = [sample["seq"] for sample in samples]
    metrics = calculate(seqs, native)
    if not args.skip_perplexity and samples:
        perplexities = [sample["pseudo_perplexity"] for sample in samples]
        metrics["pseudo_perplexity_mean"] = sum(perplexities) / len(perplexities)
        metrics["pseudo_perplexity_min"] = min(perplexities)
        metrics["pseudo_perplexity_max"] = max(perplexities)

    selected_lengths = [sample["length"] for sample in samples]
    distinct_lengths = sorted(set(selected_lengths))
    gap_display = "/".join(map(str, distinct_lengths)) or "—"
    print_callback(name, pt1, pt2, gap_display, native, args, samples, metrics)

    return {
        "name": name,
        "pt1": pt1,
        "pt2": pt2,
        "target_length": distinct_lengths[0] if len(distinct_lengths) == 1 else None,
        "generated_lengths": selected_lengths,
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
