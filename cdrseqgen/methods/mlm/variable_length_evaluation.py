"""Held-out evaluation for bounded or inferred-length MLM generation."""

import json
import logging
import statistics

from cdrseqgen.data.dataset import framework_for_dataset
from cdrseqgen.eval.metrics import calculate
from cdrseqgen.methods.mlm.generation import (
    prepare_fixed_length_prompt,
    sample_length_conditioned_spans,
)
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood

logger = logging.getLogger(__name__)


def _aggregate(records, key):
    values = [record["metrics"].get(key) for record in records]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else float("nan")


def run_variable_generative_eval(
    model,
    tokenizer,
    eval_dataset,
    *,
    device,
    length_predictor,
    length_policy,
    length_temperature,
    decoding,
    num_samples,
    temperature,
    top_k,
    top_p,
    selected,
    seed,
    skip_perplexity,
    judge_model,
    judge_tokenizer,
    judge_dataset,
    judge_name,
    output_path,
    meta,
    title,
    generator,
    log_report,
    summary_keys,
):
    """Generate without native mask count while keeping truth for scoring only."""
    records = []
    for i in selected:
        framework = framework_for_dataset(eval_dataset, i)
        native = framework["target"].upper().strip()
        generated, length_metadata = sample_length_conditioned_spans(
            model,
            tokenizer,
            device,
            generator,
            pt1=framework["pt1"],
            pt2=framework["pt2"],
            policy=length_policy,
            length_predictor=length_predictor,
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
            decoding=decoding,
            num_samples=num_samples,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            length_temperature=length_temperature,
        )
        seqs = [tokenizer.decode(span.token_ids).upper().strip() for span in generated]
        metrics = calculate(seqs, native)
        metrics["target_len"] = len(native)

        if not skip_perplexity:
            perplexities = [
                pseudo_log_likelihood(
                    model, span.input_ids, span.mask_positions, span.token_ids,
                    tokenizer.special_ids, tokenizer.mask_id, device,
                )[1]
                for span in generated
            ]
            if perplexities:
                metrics["self_perplexity_mean"] = sum(perplexities) / len(perplexities)

        if judge_model is not None:
            full_index = eval_dataset.indices[i] if hasattr(eval_dataset, "indices") else i
            judge_framework = framework_for_dataset(judge_dataset, full_index)
            judge_perplexities = []
            for sequence in seqs:
                prompt = prepare_fixed_length_prompt(
                    judge_tokenizer,
                    judge_framework["pt1"],
                    judge_framework["pt2"],
                    len(sequence),
                    partner_chain=judge_framework.get("partner_chain"),
                    antigen=judge_framework.get("antigen"),
                    max_len=judge_framework["max_len"],
                )
                candidate = judge_tokenizer.encode(sequence)
                if len(candidate) != len(prompt.mask_positions):
                    continue
                judge_perplexities.append(pseudo_log_likelihood(
                    judge_model, prompt.input_ids, prompt.mask_positions, candidate,
                    judge_tokenizer.special_ids, judge_tokenizer.mask_id, device,
                )[1])
            if judge_perplexities:
                metrics["judge_perplexity_mean"] = (
                    sum(judge_perplexities) / len(judge_perplexities)
                )

        records.append({
            "eval_index": i,
            "target_len": len(native),
            "generated_lengths": [len(sequence) for sequence in seqs],
            "length_policy": length_metadata,
            "native": native,
            "samples": seqs,
            "metrics": metrics,
        })

    log_report(records, title=title, judge_name=judge_name)
    payload = dict(meta or {})
    payload.update({
        "judge": judge_name,
        "length_policy": {
            **length_policy.as_dict(),
            "length_temperature": length_temperature,
        },
        "decoding": {
            "mode": decoding,
            "num_samples": num_samples,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
        },
        "n_eval_total": len(eval_dataset),
        "n_generated": len(records),
        "summary": {key: _aggregate(records, key) for key, _label in summary_keys},
        "records": records,
    })
    if output_path:
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("💾 Saved %d framework records to: %s", len(records), output_path)
    return payload
