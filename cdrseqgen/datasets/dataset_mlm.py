"""Masked-LM view of the antibody data: framework parts -> one masked-infilling example.

MaskedInfillingDataset:
  * __init__                -- constructor; adds masking_policy + mixed_full_span_probability
                               to the base dataset's arguments
  * masking_view            -- same records, another corruption policy (evaluation reuses the
                               loaded corpus instead of reading it again)
  * _masked_target_offsets  -- which target offsets this example hides and supervises
  * __getitem__             -- mask the loop, inject context, cap length, build labels

Utility functions:
  * normalize_masking_policy    -- canonical policy name, or a useful error
  * assemble_masked_input       -- [cls] pt1 [mask x n] pt2 [end]
  * mlm_collate_fn              -- pad a batch and derive its attention mask

Notes:
  * Two policies (MASKING_POLICIES): `full_span` hides the whole loop, matching how fixed-length
    generation is evaluated. `mixed` hides a uniformly random count instead, leaving at least one
    residue revealed and one masked, so training sees the partially filled states the iterative
    decoder actually passes through.
  * The mask count is drawn per __getitem__ call, so an example's corruption differs between
    epochs. That is deliberate and unlike context dropout, which is fixed per example — compare
    `_keep_context` in dataset_common.
  * This view supplies the loop's length by construction: it inserts exactly `len(target)` mask
    tokens, so the model is told how long the answer is. Length-free generation is a separate
    protocol (the length head, or AR/S2S `<eos>`), and recovery numbers from the two are not
    comparable.
"""
import copy

import torch
from torch.utils.data import DataLoader

from cdrseqgen.datasets.dataset_common import (
    AntibodyFrameworkDataset,
    AntibodyTokenizer,
    TokenizerAdapter,
    framework_for_dataset,
    inject_context,
    pad_batch, length_mask)

MASKING_POLICIES = ("full_span", "mixed")


def normalize_masking_policy(policy):
    """Return the canonical training masking policy name or raise a useful error."""
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "fullspan":
        normalized = "full_span"
    if normalized not in MASKING_POLICIES:
        raise ValueError(
            f"masking_policy must be one of {MASKING_POLICIES}, got {policy!r}"
        )
    return normalized


def assemble_masked_input(tokenizer, pt1_ids, pt2_ids, n_masked):
    """Lay out one masked-infilling input: [cls] pt1 [mask x n_masked] pt2 [end].

    Returns (input_ids, mask_start_idx). The wrap tokens are dropped for tokenizers that don't use
    them (the scratch model), which also shifts mask_start_idx by one.
    """
    mask_ids = [tokenizer.mask_id] * n_masked
    if tokenizer.uses_special_tokens:
        input_ids = [tokenizer.cls_id] + pt1_ids + mask_ids + pt2_ids + [tokenizer.end_id]
        return input_ids, 1 + len(pt1_ids)   # 1 for the leading cls token
    return pt1_ids + mask_ids + pt2_ids, len(pt1_ids)


class MaskedInfillingDataset(AntibodyFrameworkDataset):
    """One processed entry -> a masked-infilling example for the chosen CDR loop.

    The framework/target split, context injection and length capping come from
    :class:`~cdrseqgen.datasets.dataset_common.AntibodyFrameworkDataset`. This subclass adds only what is
    specific to masked training: which loop offsets are hidden, and the label tensor that scores
    them.

    Args (beyond the base):
      masking_policy              -- full_span (default) or mixed training corruption
      mixed_full_span_probability -- fraction of mixed examples that still hide the entire loop
    """

    def __init__(self, json_path, tokenizer, max_len=256, target_region="h3",
                 include_partner_chain=False, include_antigen=False,
                 context_dropout=0.0, context_dropout_seed=0,
                 masking_policy="full_span", mixed_full_span_probability=0.5):
        super().__init__(
            json_path, tokenizer, max_len=max_len, target_region=target_region,
            include_partner_chain=include_partner_chain, include_antigen=include_antigen,
            context_dropout=context_dropout, context_dropout_seed=context_dropout_seed,
        )
        self.masking_policy = normalize_masking_policy(masking_policy)
        if not 0.0 <= mixed_full_span_probability <= 1.0:
            raise ValueError("mixed_full_span_probability must be between 0 and 1")
        self.mixed_full_span_probability = mixed_full_span_probability

    def masking_view(self, masking_policy, mixed_full_span_probability=None):
        """Return a lightweight dataset view sharing records but using another corruption policy."""
        view = copy.copy(self)
        view.masking_policy = normalize_masking_policy(masking_policy)
        if mixed_full_span_probability is not None:
            if not 0.0 <= mixed_full_span_probability <= 1.0:
                raise ValueError("mixed_full_span_probability must be between 0 and 1")
            view.mixed_full_span_probability = mixed_full_span_probability
        return view

    def _masked_target_offsets(self, target_len):
        """Choose target offsets hidden and supervised in this training example."""
        if (
            self.masking_policy == "full_span"
            or target_len <= 1
            or torch.rand(()).item() < self.mixed_full_span_probability
        ):
            return list(range(target_len))

        # A non-full mixed example reveals at least one and masks at least one residue. Sampling
        # the count uniformly exposes the iterative decoder's full range of partially filled states.
        masked_count = int(torch.randint(1, target_len, (1,)).item())
        return sorted(torch.randperm(target_len)[:masked_count].tolist())

    def __getitem__(self, idx):
        """Assemble one example: mask the target loop, inject context, cap length, build labels.

        Returns {input_ids, labels, target_offsets, target_len, masked_count}. `labels` carries native
        residues only at positions hidden in this view and -100 everywhere else. `target_offsets`
        aligns each sequence position to its original loop offset (-1 outside the loop), which lets
        losses classify anchors/core even under mixed masking. `target_len` remains the true loop
        length; evaluation uses a full-span view, so its mask count still equals that length.
        """
        framework = self.framework_parts(idx)
        pt1_seq, target_seq, pt2_seq = framework["pt1"], framework["target"], framework["pt2"]

        # 2. Tokenize target-chain parts (adapter.encode never adds special tokens)
        t = self.tokenizer
        pt1_ids = t.encode(pt1_seq)
        pt2_ids = t.encode(pt2_seq)
        target_ids = t.encode(target_seq)

        minimum = len(target_ids) + (2 if t.uses_special_tokens else 0)
        if self.max_len < minimum:
            raise ValueError(
                f"max_len={self.max_len} cannot hold the {len(target_ids)}-token target "
                f"and its wrappers; raise max_len to >= {minimum}. "
                "Only context may be truncated, not the supervised target.")

        # --- CONTEXT INJECTION + LENGTH CAPPING --- (shared with generate.py; see inject_context)
        pt1_ids, pt2_ids = inject_context(
            t, pt1_ids, pt2_ids, len(target_ids),
            partner_seq=framework["partner_chain"], antigen_seq=framework["antigen"],
            max_len=self.max_len,
        )

        # 3. Construct Final Input and Find Mask Start
        input_ids, mask_start_idx = assemble_masked_input(t, pt1_ids, pt2_ids, len(target_ids))

        # 4. Optionally reveal native training residues. Evaluation always uses a separate
        # full_span view (pipeline.py), so no native target residue leaks into reported recovery.
        masked_offsets = set(self._masked_target_offsets(len(target_ids)))

        # 5. Create labels only for residues the model must reconstruct. Target offsets remain for
        # every loop position, including visible mixed-mask residues, so a custom loss can still
        # distinguish core from anchors when the supervised positions are non-contiguous.
        labels = [-100] * len(input_ids)
        target_offsets = [-1] * len(input_ids)
        for i, t_id in enumerate(target_ids):
            position = mask_start_idx + i
            target_offsets[position] = i
            if i in masked_offsets:
                labels[position] = t_id
            else:
                input_ids[position] = t_id

        assert len(input_ids) == len(labels), f"Length Mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_offsets": torch.tensor(target_offsets, dtype=torch.long),
            "target_len": len(target_ids),
            "masked_count": len(masked_offsets),
        }



def mlm_collate_fn(batch, pad_token_id=0):
    """Pad a batch to its longest example.

    [NOTE] pad_token_id defaults to 0, which is correct for AntibodyTokenizer's <pad> id, but
    NOT for ESM-2's tokenizer (whose <pad> id is 1, while 0 is <cls>) — callers using an
    ESM-2 tokenizer must pass its real pad_token_id explicitly (see train.py/generate.py).
    That id reaches the padding value only; the attention mask comes from each example's
    length, so a pad id that collides with a residue cannot mask out real tokens.
    """
    input_ids = pad_batch(batch, "input_ids", pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": pad_batch(batch, "labels", -100),      # -100 is ignored by the loss
        "attention_mask": length_mask(input_ids, batch, "input_ids"),
        "target_offsets": pad_batch(batch, "target_offsets", -1),
        "target_lengths": torch.tensor([item["target_len"] for item in batch], dtype=torch.long),
    }
