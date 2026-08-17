"""Shared fixed, bounded, and inferred-length candidate generation.

Callers resolve a length policy first, then this module assembles the selected number of masks and
fills them with either masked-LM decoder. Keeping prompt assembly and decoder dispatch here prevents
the demo, held-out evaluation, and benchmark adapter from quietly implementing different inference
behavior. Variable policies select lengths first, group equal-length draws, then reuse the fixed span.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cdrseqgen.data.dataset import assemble_masked_input, inject_context
from cdrseqgen.methods.mlm.length import LengthPolicy, fixed_length
from cdrseqgen.methods.mlm.sampling import sample_independent, sample_iterative


@dataclass(frozen=True)
class FixedLengthPrompt:
    """One assembled masked-LM prompt plus auditable truncation information."""

    input_ids: list[int]
    mask_positions: list[int]
    target_length: int
    max_len: int
    pt1_tokens: int
    pt2_tokens: int
    pt1_truncated_tokens: int
    pt2_truncated_tokens: int

    def metadata(self) -> dict[str, int]:
        """JSON-ready prompt metadata used by the benchmark adapter."""
        return {
            "max_len": self.max_len,
            "input_tokens": len(self.input_ids),
            "pt1_tokens": self.pt1_tokens,
            "pt2_tokens": self.pt2_tokens,
            "pt1_truncated_tokens": self.pt1_truncated_tokens,
            "pt2_truncated_tokens": self.pt2_truncated_tokens,
        }


def prepare_fixed_length_prompt(
    tokenizer,
    pt1: str,
    pt2: str,
    target_length: int,
    *,
    partner_chain: str | None = None,
    antigen: str | None = None,
    max_len: int = 256,
) -> FixedLengthPrompt:
    """Encode context and assemble ``target_length`` masks between the two framework flanks."""
    target_length = fixed_length(target_length, source="prompt_argument").length
    pt1_ids = tokenizer.encode(pt1)
    pt2_ids = tokenizer.encode(pt2)
    original_pt1_len = len(pt1_ids)
    original_pt2_len = len(pt2_ids)
    pt1_ids, pt2_ids = inject_context(
        tokenizer,
        pt1_ids,
        pt2_ids,
        target_length,
        partner_seq=partner_chain,
        antigen_seq=antigen,
        max_len=max_len,
    )
    input_ids, mask_start = assemble_masked_input(
        tokenizer, pt1_ids, pt2_ids, target_length
    )
    return FixedLengthPrompt(
        input_ids=input_ids,
        mask_positions=list(range(mask_start, mask_start + target_length)),
        target_length=target_length,
        max_len=max_len,
        pt1_tokens=len(pt1_ids),
        pt2_tokens=len(pt2_ids),
        pt1_truncated_tokens=original_pt1_len - len(pt1_ids),
        pt2_truncated_tokens=original_pt2_len - len(pt2_ids),
    )


def sample_masked_span(
    model,
    tokenizer,
    device,
    generator,
    input_ids: list[int],
    mask_positions: list[int],
    *,
    decoding: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> list[list[int]]:
    """Fill one assembled masked span with the requested existing decoder."""
    kwargs: dict[str, Any] = {
        "model": model,
        "input_ids": input_ids,
        "mask_positions": mask_positions,
        "banned_ids": tokenizer.special_ids,
        "device": device,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "generator": generator,
    }
    if decoding == "independent":
        return sample_independent(**kwargs)
    return sample_iterative(**kwargs, mask_id=tokenizer.mask_id)


@dataclass(frozen=True)
class GeneratedSpan:
    """One sampled loop plus the exact prompt needed for architecture-native scoring."""

    token_ids: list[int]
    length: int
    input_ids: list[int]
    mask_positions: list[int]
    prompt_metadata: dict[str, int]


def sample_length_conditioned_spans(
    model,
    tokenizer,
    device,
    generator,
    *,
    pt1: str,
    pt2: str,
    policy: LengthPolicy,
    length_predictor=None,
    partner_chain: str | None = None,
    antigen: str | None = None,
    max_len: int = 256,
    decoding: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
    length_temperature: float = 1.0,
) -> tuple[list[GeneratedSpan], dict]:
    """Select candidate lengths, expand masks, and sample residues for every draw."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    actual_samples = 1 if temperature == 0 else num_samples
    probabilities = None
    gap_prompt = None
    if policy.requires_predictor:
        if length_predictor is None:
            raise FileNotFoundError(
                "bounded/inferred generation requires this checkpoint's trained length predictor"
            )
        lengths, probabilities, gap_prompt = length_predictor.select(
            model,
            tokenizer,
            device,
            generator,
            pt1=pt1,
            pt2=pt2,
            policy=policy,
            num_samples=actual_samples,
            partner_chain=partner_chain,
            antigen=antigen,
            max_len=max_len,
            temperature=length_temperature,
        )
    else:
        lengths = [policy.length] * actual_samples

    positions_by_length = defaultdict(list)
    for output_index, length in enumerate(lengths):
        positions_by_length[length].append(output_index)
    results = [None] * actual_samples
    prompts = {}
    for length, output_positions in positions_by_length.items():
        prompt = prepare_fixed_length_prompt(
            tokenizer,
            pt1,
            pt2,
            length,
            partner_chain=partner_chain,
            antigen=antigen,
            max_len=max_len,
        )
        prompts[str(length)] = prompt.metadata()
        sampled = sample_masked_span(
            model,
            tokenizer,
            device,
            generator,
            prompt.input_ids,
            prompt.mask_positions,
            decoding=decoding,
            num_samples=len(output_positions),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        for output_index, token_ids in zip(output_positions, sampled):
            results[output_index] = GeneratedSpan(
                token_ids=token_ids,
                length=length,
                input_ids=prompt.input_ids,
                mask_positions=prompt.mask_positions,
                prompt_metadata=prompt.metadata(),
            )

    metadata = policy.as_dict(selected_lengths=lengths, probabilities=probabilities)
    metadata["length_temperature"] = length_temperature
    metadata["prompts_by_length"] = prompts
    if gap_prompt is not None:
        metadata["gap_prompt"] = {
            "input_tokens": len(gap_prompt.input_ids),
            "gap_position": gap_prompt.gap_position,
            "reserve_length": gap_prompt.reserve_length,
        }
    return results, metadata
