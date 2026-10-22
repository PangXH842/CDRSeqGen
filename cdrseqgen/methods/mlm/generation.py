"""Own the MLM generation workflow and reusable prompt/sampling helpers.

Command workflow:
  generate.py -> gen_cli.main -> run_generation -> common.generation lifecycle
  The shared lifecycle calls prepare_generation, generation_state and generate_case.

Workflow functions:
  * run_generation     -- master command workflow using the shared loading/output lifecycle
  * prepare_generation -- validate options and resolve one length policy per case
  * generation_state   -- load the length predictor when any case requires it
  * generate_case      -- resolve a case and delegate its scored record to case_generation
  * resolve_case_length -- combine case/command settings and retain optional native truth

Reusable inference helpers (also used by evaluation and benchmark callers):
  * prepare_fixed_length_prompt   -- encode context and insert the selected number of masks
  * sample_masked_span            -- dispatch an assembled prompt to an MLM decoder
  * sample_length_conditioned_spans -- select lengths, build prompts and collect candidate draws

Data classes:
  * FixedLengthPrompt -- prompt tokens, mask positions and context-length metadata
  * GeneratedSpan     -- candidate tokens plus the prompt needed for self-scoring

Workflow:
  caller's length policy -> selected lengths -> group draws by length
                         -> assemble prompts -> sampling.py decoder -> GeneratedSpan objects

Sequence measurements and dataset scoring/reporting are supplied by cdrseqgen.eval.
case_generation assembles individual case records; evaluation coordinates dataset runs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from cdrseqgen.datasets.dataset_mlm import assemble_masked_input, inject_context
from cdrseqgen.methods.mlm.length import LengthPolicy, fixed_length, resolve_length_policy
from cdrseqgen.methods.mlm.length_predictor import LengthPredictor
from cdrseqgen.methods.mlm.sampling import sample_independent, sample_iterative


def run_generation(args, config):
    """Run the generation command with MLM's length policies and per-case producer.

    common.generation handles compatibility, loading, seeding and output.
    This module supplies preparation, optional length-head loading and generation.
    """
    from cdrseqgen.methods.common.generation import run_generation_pipeline

    return run_generation_pipeline(
        "mlm", args, config,
        prepare_generation=prepare_generation,
        generation_state=generation_state,
        generate_case=generate_case,
    )


def resolve_case_length(case, args):
    """Return normalized native truth and the case's generation length policy.

    Case-level settings override command settings. Native truth is retained for
    scoring and supplies generation length only when use_native_length is explicit.
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
    """Validate MLM command options and return one length policy per supplied case.

    Sets the existing iterative decoder default in args. config is accepted as
    part of the shared method-command interface; MLM does not use it here.
    """
    if getattr(args, 'max_new_tokens', None) is not None:
        raise ValueError('--max_new_tokens applies to autoregressive decoding, not masked infilling')
    args.decoding = args.decoding or 'iterative'  # preserve the qualitative tool's existing default
    if args.decoding == 'causal':
        raise ValueError('Masked infilling requires independent or iterative decoding')
    return [resolve_case_length(case, args)[1] for case in cases]


def generation_state(args, policies, device):
    """Load the saved length predictor if required by any case; otherwise return None."""
    return LengthPredictor.load(args.model_path, device=device) if any(p.requires_predictor for p in policies) else None


def generate_case(model, tokenizer, device, args, generator, case, state, print_callback):
    """Resolve one case and return its generated/scored JSON-ready record.

    state is the optional predictor loaded by generation_state(). The record
    builder is imported on demand because it also uses this module's span helpers.
    """
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


@dataclass(frozen=True)
class FixedLengthPrompt:
    """Store an assembled prompt, its mask positions and context-length metadata."""

    input_ids: list[int]
    mask_positions: list[int]
    target_length: int
    max_len: int
    pt1_tokens: int
    pt2_tokens: int
    pt1_truncated_tokens: int
    pt2_truncated_tokens: int

    def metadata(self) -> dict[str, int]:
        """Return JSON-ready token counts used in generation and benchmark metadata."""
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
    """Return a prompt with target_length masks between the framework flanks.

    Example: pt1="AC", pt2="FG", target_length=3, no extra context
      prompt: A C [mask] [mask] [mask] F G    (plus tokenizer wrapper tokens, if used)

    Dataset helpers place optional partner/antigen context and apply max_len
    truncation rules. The supplied target_length is validated, not inferred here.
    """
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
    """Return candidate token-ID lists for an already assembled masked prompt.

    Dispatch:
      decoding="independent" -> sample_independent
      otherwise              -> sample_iterative

    The caller validates the decoder name. Both decoders exclude tokenizer.special_ids
    and return one greedy candidate when temperature=0, regardless of num_samples.
    """
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
    """Store one sampled loop, its selected length and the prompt used to generate it.

    input_ids and mask_positions let evaluation score the candidate in the same
    context; prompt_metadata records the assembled prompt's token counts.
    """

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
    """Return (generated spans, length metadata) under a fixed, bounded or inferred policy.

    Steps:
      1. Use the fixed length, or ask length_predictor to select one length per draw.
      2. Group draws with the same length and build one masked prompt per group.
      3. Sample residues and place each GeneratedSpan back in its original draw order.

    Example:
      selected lengths: [8, 10, 8] -> prompt groups: {8: [0, 2], 10: [1]}
      returned spans:   [length 8, length 10, length 8]

    Notes:
      * No native loop or native length is required by this interface.
      * Bounded/inferred policies require the checkpoint's fitted length predictor.
      * temperature=0 produces one draw; length_temperature separately controls
        length selection by the predictor.
      * Metadata records the policy, selected lengths and prompt counts by length.
    """
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
