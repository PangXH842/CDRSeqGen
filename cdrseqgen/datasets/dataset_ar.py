"""Autoregressive view: framework/context -> prompt, then target + terminator.

Default layout (pt1 and pt2 include any retained partner/antigen context):

    input_ids:  pt1  <mask>  pt2  <sep>  target  <eos>
    labels:     -100 across prompt     target  <eos>
                |------ prompt ------| |-- answer --|

CausalInfillingDataset:             [*] from AntibodyFrameworkDataset template
  * __init__                      -- load records and validate the adapter's layout
  * __getitem__
    * [*] framework_parts         -- raw framework/target/context strings for this entry
    * build_causal_prompt         -- inject/cap context, then apply the layout
    * build_layout_answer         -- target tokens followed by one terminator

CausalTokenizer:
  * the shared 25-token vocab with <bos>/<eos> appended at 25/26, so ids 0-24 keep
    their meaning and every existing MLM checkpoint's embedding table still matches

SpanLayout:
  * the frozen token-arrangement contract — id/version, n_layout_tokens, and three
    builder callbacks. See the class for the full signature.

Utility functions:
  * causal_adapter                  -- TokenizerAdapter exposing bos_id/eos_id, with the
                                       default layout attached and validated
  * CDRSEQGEN_CAUSAL                -- the shipped layout: pt1 <mask> pt2 <sep> | target <eos>
  * validate_layout                 -- check metadata and representative examples
  * span_layout_metadata            -- contract recorded with checkpoints
  * causal_collate_fn               -- right-pad tensors and build attention masks
  * validate_causal_decode_steps    -- bound sampling by the fixed answer budget

Notes:
  * Training and generation share build_causal_prompt. It never reads the native
    target or target length: both paths reserve the same 96 answer-token positions.
  * The layout counts its prompt markers plus the answer terminator. The default
    spends three positions: <mask>, <sep>, <eos>; it does not insert <bos>.
  * Context may be truncated; the training answer must fit intact. Increasing the
    reserve can remove context, so budget max_len for the desired context as well.
  * Layout builders receive capped flank copies and public options, not the raw
    framework. Only target-token + terminator answers are currently supported;
    other output formats would also need changes to the sampler's extraction.
"""
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from cdrseqgen.datasets.dataset_common import (
    AntibodyFrameworkDataset,
    AntibodyTokenizer,
    TokenizerAdapter,
    inject_context,
    length_mask,
    pad_batch,
    validate_decode_steps,
)


BOS = "<bos>"
EOS = "<eos>"


class CausalTokenizer(AntibodyTokenizer):
    """The shared vocab, extended with <bos>/<eos> at the end so ids 0-24 keep their meaning."""

    def __init__(self):
        super().__init__()
        for token in (BOS, EOS):
            if token in self.tok_to_id:                       # defensive: never renumber
                raise ValueError(f"{token} already in the shared vocab; ids would shift")
            self.tok_to_id[token] = len(self.vocab)
            self.id_to_tok[len(self.vocab)] = token
            self.vocab.append(token)
        self.bos_id = self.tok_to_id[BOS]
        self.eos_id = self.tok_to_id[EOS]


# The backbone attaches one layout to its adapter, shared by dataset and sampler.
@dataclass(frozen=True)
class SpanLayout:
    """Immutable token-layout contract shared by training and generation.

    id/version identify checkpoint semantics; bump version when the format changes.
    n_layout_tokens counts prompt overhead PLUS one answer terminator. For example,
    the default has two prompt markers and one terminator, so this count is three.

    build_prompt takes (adapter, capped_left_ids, capped_right_ids, public_options),
    where public_options currently contains only max_len. build_answer must return
    the supplied target IDs in order, followed by terminator_id(adapter). These
    callbacks arrange tokens; they do not tokenize, truncate context or make labels.
    """

    id: str
    n_layout_tokens: int
    build_prompt: Callable[[TokenizerAdapter, list[int], list[int], Mapping[str, int]], list[int]]
    build_answer: Callable[[TokenizerAdapter, list[int]], list[int]]
    terminator_id: Callable[[TokenizerAdapter], int]
    version: int = 1


def _cdrseqgen_prompt(t, pt1_ids, pt2_ids, options):
    """Put both flanks before the answer boundary; no BOS or outer wrappers."""
    return pt1_ids + [t.mask_id] + pt2_ids + [t.chain_sep_id]


def _cdrseqgen_answer(t, target_ids):
    """Append EOS without shifting or changing the target tokens."""
    return list(target_ids) + [t.eos_id]


# `pt1 <mask> pt2 <sep> target <eos>` -- both flanks in the prompt, loss on the answer only. Three
# layout tokens: the gap marker, the prompt/answer separator, and the terminator.
CDRSEQGEN_CAUSAL = SpanLayout(
    id="cdrseqgen_causal",
    n_layout_tokens=3,
    build_prompt=_cdrseqgen_prompt,
    build_answer=_cdrseqgen_answer,
    terminator_id=lambda t: t.eos_id,
)


def _validate_layout_header(layout, tokenizer):
    """Validate layout identity, counts and terminator; return the terminator ID."""
    if not isinstance(layout, SpanLayout):
        raise ValueError("span layout must be a SpanLayout")
    if not isinstance(layout.id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", layout.id):
        raise ValueError("span layout needs a stable nonempty id")
    for name in ("version", "n_layout_tokens"):
        value = getattr(layout, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"span layout {name} must be a positive integer")
    terminator = layout.terminator_id(tokenizer)
    if (isinstance(terminator, bool) or not isinstance(terminator, int)
            or not 0 <= terminator < tokenizer.vocab_size
            or terminator not in tokenizer.special_ids
            or terminator in tokenizer.encode("ACDEFGHIKLMNPQRSTVWY")):
        raise ValueError("span layout terminator must be an in-vocabulary non-residue special token")
    return terminator


def _checked_tokens(tokens, tokenizer, label):
    """Materialize builder output and reject non-integer or out-of-vocabulary IDs."""
    tokens = list(tokens)
    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < tokenizer.vocab_size
           for x in tokens):
        raise ValueError(f"span layout {label} contains invalid token ids")
    return tokens


def build_layout_answer(layout, tokenizer, target_ids):
    """Return target IDs followed by one terminator, or reject the layout's answer.

    Called for both setup probes and real training targets. Copy target_ids before
    invoking the callback so mutation cannot hide reordered or altered targets.
    Layout metadata/terminator validity is checked separately by validate_layout.
    """
    target = list(target_ids)
    answer = _checked_tokens(layout.build_answer(tokenizer, list(target)), tokenizer, "answer")
    if answer != target + [layout.terminator_id(tokenizer)]:
        raise ValueError(
            f"span layout {layout.id!r} builds an answer the causal sampler cannot read back: "
            "expected target residues followed by exactly one terminator. Wrapped/reordered "
            "answers need their own extraction step before they can be used.")
    return answer


def _build_layout_prompt(layout, tokenizer, pt1_ids, pt2_ids, max_len):
    """Build from capped flanks and verify token IDs, nonemptiness and overhead.

    One layout token belongs to the answer terminator, so the prompt must contain
    len(left) + len(right) + n_layout_tokens - 1 tokens. An empty prompt cannot
    supply the position whose logits predict the first answer token.
    """
    # Context is already injected and capped. Native target/length and uncapped strings are
    # deliberately absent; public options are immutable. Builders cannot mutate our flanks.
    expected = len(pt1_ids) + len(pt2_ids) + layout.n_layout_tokens - 1
    prompt = _checked_tokens(layout.build_prompt(
        tokenizer, list(pt1_ids), list(pt2_ids), MappingProxyType({"max_len": max_len})),
        tokenizer, "prompt")
    if not prompt:
        raise ValueError(f"span layout {layout.id!r} must provide at least one prompt token "
                         "after capping so the causal sampler can predict the first answer token")
    if len(prompt) != expected:
        raise ValueError(f"span layout {layout.id!r}: declared n_layout_tokens="
                         f"{layout.n_layout_tokens}, but actual prompt overhead is "
                         f"{len(prompt) - len(pt1_ids) - len(pt2_ids)} (plus one answer terminator)")
    return prompt


def validate_layout(layout, tokenizer):
    """Check the layout at setup and return it, raising on unsupported contracts.

    Probe empty and varied targets/flanks to catch wrappers, reordering and wrong
    overhead. These probes are not exhaustive: prompt and answer helpers also
    check every real example when the dataset or sampler constructs it.
    """
    _validate_layout_header(layout, tokenizer)
    for residues in ("", "A", "ACD", "ACDEFGHIKLMNPQRSTVWY"):
        build_layout_answer(layout, tokenizer, tokenizer.encode(residues))
    for left, right in (("", ""), ("AC", "D"), ("", "EFG")):
        _build_layout_prompt(layout, tokenizer, tokenizer.encode(left), tokenizer.encode(right), 384)
    return layout


def span_layout_metadata(tokenizer):
    """Return the validated ID, version, overhead and terminator for checkpoint checks."""
    layout = validate_layout(layout_for(tokenizer), tokenizer)
    return {"id": layout.id, "version": layout.version,
            "n_layout_tokens": layout.n_layout_tokens,
            "terminator_id": layout.terminator_id(tokenizer)}


def layout_for(tokenizer):
    """The layout a tokenizer adapter carries, defaulting to CDRSeqGen's own.

    An adapter built before layouts existed, or by a backbone that does not set one, behaves
    exactly as it did. Only missing/None selects the default; invalid attached
    values must reach validation rather than silently selecting another layout.
    """
    layout = getattr(tokenizer, "span_layout", None)
    return CDRSEQGEN_CAUSAL if layout is None else layout


def causal_adapter(tokenizer=None, position_type="absolute"):
    """Adapt a CausalTokenizer and attach the validated default span layout.

    Expose BOS/EOS in addition to the shared tokenizer interface. The default
    prompt uses mask_id and chain_sep_id but no outer CLS/EOS wrappers; BOS is
    available in the vocabulary without being inserted. The generic adapter may
    omit a mask token, but this particular layout requires one.
    """
    tok = tokenizer or CausalTokenizer()
    adapter = TokenizerAdapter(
        encode=tok.encode,
        decode=tok.decode,
        vocab_size=len(tok.vocab),
        pad_id=tok.pad_id,
        mask_id=tok.mask_id,
        cls_id=None,
        end_id=None,
        chain_sep_id=tok.tok_to_id["<sep>"],
        special_ids={tok.tok_to_id[t]
                     for t in ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>", BOS, EOS)},
        uses_special_tokens=False,
        position_type=position_type,
    )
    adapter.bos_id = tok.bos_id
    adapter.eos_id = tok.eos_id
    # Set on the adapter, not passed per call -- see SpanLayout on why that distinction matters.
    adapter.span_layout = validate_layout(CDRSEQGEN_CAUSAL, adapter)
    return adapter


# Kept for callers that need the default overhead; custom layouts own their count.
LAYOUT_SPECIAL_TOKENS = CDRSEQGEN_CAUSAL.n_layout_tokens

# Fixed across training and generation, independent of native length or decode settings.
# 96 leaves headroom over the recorded 63-residue maximum. With the default layout,
# max_len must be at least 96 + 3 = 99; every retained context token needs more room.
LOOP_LENGTH_RESERVE = 96
CAUSAL_DECODE_STEPS = LOOP_LENGTH_RESERVE + 1  # final step may emit the terminator


def validate_causal_decode_steps(max_new_tokens):
    """Return a decode cap that fits the prompt's fixed answer budget."""
    return validate_decode_steps(max_new_tokens, CAUSAL_DECODE_STEPS,
                                 f"prompts reserve {LOOP_LENGTH_RESERVE} residues plus one <eos> step")


def build_causal_prompt(tokenizer, framework):
    """Return (prompt_ids, metadata) for training or generation.

    framework supplies pt1, pt2, partner_chain, antigen and max_len. It may contain
    target/target_length, but neither is read here or passed to the layout builder.
    inject_context caps the flanks after reserving the fixed answer and layout
    overhead. Its existing tokenizer-wrapper allowance is also retained.

    Metadata reports retained flank token counts (including injected context) and
    the answer reserve. Raise if the budget cannot hold the layout and reserve,
    or if the layout emits invalid tokens, wrong overhead or an empty prompt.
    """
    t = tokenizer
    layout = layout_for(t)
    _validate_layout_header(layout, t)
    max_len = framework["max_len"]
    reserved = LOOP_LENGTH_RESERVE + layout.n_layout_tokens
    if max_len < reserved:
        raise ValueError(
            f"max_len={max_len} cannot hold the {LOOP_LENGTH_RESERVE}-token answer reserve "
            f"plus {layout.n_layout_tokens} layout tokens for {layout.id!r}; "
            f"raise max_len to >= {reserved}.")

    pt1_ids, pt2_ids = inject_context(
        t, t.encode(framework["pt1"]), t.encode(framework["pt2"]), reserved,
        partner_seq=framework["partner_chain"], antigen_seq=framework["antigen"],
        max_len=max_len,
    )
    prompt = _build_layout_prompt(layout, t, pt1_ids, pt2_ids, max_len)
    required = len(prompt) + CAUSAL_DECODE_STEPS
    if required > max_len:
        raise ValueError(
            f"span layout {layout.id!r}: prompt plus answer reserve and terminator "
            f"needs {required} positions, over max_len={max_len}.")
    return prompt, {
        "pt1_len": len(pt1_ids), "pt2_len": len(pt2_ids),
        "reserve": LOOP_LENGTH_RESERVE,
    }


class CausalInfillingDataset(AntibodyFrameworkDataset):
    """Processed records -> causal examples, using the base dataset's arguments.

    Inherit target selection and context dropout from AntibodyFrameworkDataset.
    Validate the adapter's layout at construction; tokenize/cap each example when
    indexed. Only the target and its terminator receive labels other than -100.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_layout(layout_for(self.tokenizer), self.tokenizer)

    def __getitem__(self, idx):
        """Return input_ids/labels tensors and integer prompt/target/flank lengths.

        Labels are unshifted; the causal model handles the one-position loss shift.
        target_len excludes the terminator. pt1_len counts the capped left context,
        including any injected context; a custom layout may place it differently.
        """
        framework = self.framework_parts(idx)
        t = self.tokenizer
        target_ids = t.encode(framework["target"])
        if len(target_ids) > LOOP_LENGTH_RESERVE:
            raise ValueError(
                f"loop of {len(target_ids)} residues exceeds the {LOOP_LENGTH_RESERVE}-position "
                "reserve, so the assembled example could overflow max_len. "
                "Raise LOOP_LENGTH_RESERVE.")
        prompt, meta = build_causal_prompt(t, framework)
        answer = build_layout_answer(layout_for(t), t, target_ids)

        input_ids = prompt + answer
        if len(input_ids) > self.max_len:
            raise ValueError(
                f"assembled causal example is {len(input_ids)} tokens, over "
                f"max_len={self.max_len}: a {len(prompt)}-token prompt plus a "
                f"{len(answer)}-token answer. The flanks are capped before this point, so "
                "raise max_len.")
        labels = [-100] * len(prompt) + answer      # the prompt is given, not predicted

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt),
            "target_len": len(target_ids),
            "pt1_len": meta["pt1_len"],
        }


def causal_collate_fn(batch, pad_token_id=0):
    """Right-pad training examples; return only input_ids, labels, attention_mask.

    Token padding uses pad_token_id; label padding uses -100 to exclude it from
    loss. Attention comes from actual lengths, even if a real token equals the
    padding ID. Per-example metadata is omitted from the model inputs. Generation
    constructs its own prompt batches rather than using this training collator.
    """
    input_ids = pad_batch(batch, "input_ids", pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": pad_batch(batch, "labels", -100),
        "attention_mask": length_mask(input_ids, batch, "input_ids"),
    }
