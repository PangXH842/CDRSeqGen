"""Generate and score one framework case with the masked-LM method."""

from cdrseqgen.eval.metrics import amino_acid_recovery, calculate
from cdrseqgen.methods.mlm.generation import sample_length_conditioned_spans
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood


def generate_case_record(
    model,
    tokenizer,
    device,
    args,
    generator,
    case,
    *,
    native,
    length_policy,
    length_predictor,
    print_callback,
):
    """Generate, score, and serialize one fixed, bounded, or inferred-length case."""
    name = case.get("name", "Unnamed Test")
    pt1, pt2 = case["pt1"], case["pt2"]
    partner = case.get("partner_chain") or args.partner_chain
    antigen = case.get("antigen") or args.antigen

    # One call for every policy: sample_length_conditioned_spans already branches on
    # policy.requires_predictor, so a fixed length is just the case where the length is known
    # before the prompt is built rather than predicted from it.
    generated, length_metadata = sample_length_conditioned_spans(
        model,
        tokenizer,
        device,
        generator,
        pt1=pt1,
        pt2=pt2,
        policy=length_policy,
        length_predictor=length_predictor,
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
    gap_display = (
        str(selected_lengths[0]) if len(set(selected_lengths)) == 1
        else "/".join(str(length) for length in sorted(set(selected_lengths)))
    ) if selected_lengths else "—"
    print_callback(name, pt1, pt2, gap_display, native, args, samples, metrics)

    return {
        "name": name,
        "pt1": pt1,
        "pt2": pt2,
        "target_length": selected_lengths[0] if len(set(selected_lengths)) == 1 else None,
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
