"""Configurable training losses for masked-LM CDR reconstruction.

All MLM backbones reach this module through :class:`MLMLossTrainer`, so adding a loss does not
require editing the scratch transformer and every Hugging Face model separately.

Supported losses:
  * cross_entropy            -- the original token-averaged MLM objective
  * weighted_cross_entropy   -- cross-entropy with configurable core/anchor token weights

The dataset supplies target offsets and lengths as training metadata. The trainer removes those
fields before invoking the model, then the selected loss uses them if needed. This preserves one
model-forward contract across scratch, ESM-2, IgBert, and ProtBert, including mixed masking where
the supervised target positions need not be contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import Trainer


IGNORE_INDEX = -100
LOSS_TYPES = ("cross_entropy", "weighted_cross_entropy")
_LOSS_ALIASES = {
    "ce": "cross_entropy",
    "crossentropy": "cross_entropy",
    "weighted": "weighted_cross_entropy",
    "weighted_ce": "weighted_cross_entropy",
}


def normalize_loss_type(loss_type: str) -> str:
    """Return a canonical loss name or raise a configuration error."""
    normalized = str(loss_type).strip().lower().replace("-", "_")
    normalized = _LOSS_ALIASES.get(normalized, normalized)
    if normalized not in LOSS_TYPES:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
    return normalized


@dataclass(frozen=True)
class LossConfig:
    """Resolved loss settings recorded with a run and consumed by ``MLMLossTrainer``."""

    loss_type: str = "cross_entropy"
    core_weight: float = 1.0
    anchor_weight: float = 1.0
    anchor_trim: int = 2
    label_smoothing: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "loss_type", normalize_loss_type(self.loss_type))
        if self.core_weight < 0 or self.anchor_weight < 0:
            raise ValueError("core_loss_weight and anchor_loss_weight must be non-negative")
        if self.core_weight == 0 and self.anchor_weight == 0:
            raise ValueError("core_loss_weight and anchor_loss_weight cannot both be zero")
        if (
            isinstance(self.anchor_trim, bool)
            or not isinstance(self.anchor_trim, int)
            or self.anchor_trim < 0
        ):
            raise ValueError("anchor_trim must be a non-negative integer")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing_factor must be in [0, 1)")

    @property
    def requires_target_metadata(self) -> bool:
        return self.loss_type == "weighted_cross_entropy"


def loss_config_from_args(model_args, training_args=None) -> LossConfig:
    """Resolve a :class:`LossConfig` from CLI/config dataclasses."""
    return LossConfig(
        loss_type=getattr(model_args, "loss_type", "cross_entropy"),
        core_weight=float(getattr(model_args, "core_loss_weight", 1.0)),
        anchor_weight=float(getattr(model_args, "anchor_loss_weight", 1.0)),
        anchor_trim=int(getattr(model_args, "anchor_trim", 2)),
        label_smoothing=float(
            getattr(training_args, "label_smoothing_factor", 0.0)
            if training_args is not None else 0.0
        ),
    )


def _token_cross_entropy(logits, labels, *, label_smoothing, reduction):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def _cross_entropy(logits, labels, _target_offsets, _target_lengths, config):
    return _token_cross_entropy(
        logits, labels, label_smoothing=config.label_smoothing, reduction="mean"
    )


def _weighted_cross_entropy(logits, labels, target_offsets, target_lengths, config):
    if target_offsets is None or target_lengths is None:
        raise ValueError(
            "weighted_cross_entropy requires target_offsets and target_lengths from collate_fn"
        )
    if target_offsets.shape != labels.shape:
        raise ValueError(
            f"target_offsets shape {tuple(target_offsets.shape)} does not match labels "
            f"{tuple(labels.shape)}"
        )
    if target_lengths.ndim != 1 or target_lengths.size(0) != labels.size(0):
        raise ValueError("target_lengths must contain one length per batch row")

    active = labels.ne(IGNORE_INDEX)
    if not bool(active.any()):
        raise ValueError("cannot calculate MLM loss for a batch with no supervised tokens")
    valid_offsets = target_offsets[active]
    active_lengths = target_lengths[:, None].expand_as(labels)[active]
    if bool(((valid_offsets < 0) | (valid_offsets >= active_lengths)).any()):
        raise ValueError("supervised labels carry invalid target offsets")

    # A loop has a core only when something remains after trimming both ends; for short loops every
    # supervised residue is an anchor.
    #
    # This trims by TRUE loop offset, whereas recovery._core_anchor_masks trims the ends of the
    # *supervised span*. Under full_span masking the two are identical (every loop position is
    # supervised). Under mixed masking they diverge, and this one is right: recovery's version would
    # call the first supervised residue an anchor even when it sits mid-loop. That divergence is not
    # live, because evaluation always runs on a full_span view (see training.get_datasets).
    lengths = target_lengths[:, None]
    has_core = lengths > 2 * config.anchor_trim
    core = (
        active
        & has_core
        & target_offsets.ge(config.anchor_trim)
        & target_offsets.lt(lengths - config.anchor_trim)
    )
    token_weights = torch.full_like(labels, config.anchor_weight, dtype=logits.dtype)
    token_weights = torch.where(
        core,
        torch.as_tensor(config.core_weight, dtype=logits.dtype, device=logits.device),
        token_weights,
    )
    token_weights = token_weights * active.to(logits.dtype)

    token_losses = _token_cross_entropy(
        logits, labels, label_smoothing=config.label_smoothing, reduction="none"
    ).reshape_as(labels)
    total_weight = token_weights.sum()
    if not bool(total_weight > 0):
        # Reachable with a zero weight on the only class present. anchor_loss_weight=0 is a valid
        # "train on the core only" setting for H3 (1 of 21,106 loops has no core at trim 2), but on
        # a short loop like L2 -- 98.8% of which are 3 residues -- every supervised token is an
        # anchor, so the whole batch would weigh nothing. Fail here rather than emit a NaN loss.
        raise ValueError(
            f"every supervised token in this batch has weight zero (core={config.core_weight}, "
            f"anchor={config.anchor_weight}, trim={config.anchor_trim}). No loop in the batch is "
            f"longer than {2 * config.anchor_trim} residues, so there is no core to train on — "
            "raise the zero weight above 0, or lower anchor_trim for this loop."
        )
    return (token_losses * token_weights).sum() / total_weight


LOSS_REGISTRY: dict[str, Callable] = {
    "cross_entropy": _cross_entropy,
    "weighted_cross_entropy": _weighted_cross_entropy,
}


def calculate_mlm_loss(
    logits,
    labels,
    *,
    config: LossConfig,
    target_offsets=None,
    target_lengths=None,
):
    """Calculate the configured loss from common ``[batch, sequence, vocab]`` logits."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(
            f"expected logits [B,S,V] and labels [B,S], got {tuple(logits.shape)} and "
            f"{tuple(labels.shape)}"
        )
    if not bool(labels.ne(IGNORE_INDEX).any()):
        raise ValueError("cannot calculate MLM loss for a batch with no supervised tokens")
    return LOSS_REGISTRY[config.loss_type](
        logits, labels, target_offsets, target_lengths, config
    )


def _logits_of(outputs):
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[0]


class MLMLossTrainer(Trainer):
    """Hugging Face Trainer bridge that applies one shared MLM loss to every backbone."""

    LOSS_METADATA_KEYS = {"target_offsets", "target_lengths"}

    def __init__(self, *args, loss_config: LossConfig | None = None, **kwargs):
        self.loss_config = loss_config or LossConfig()
        super().__init__(*args, **kwargs)
        # Our loss is a microbatch mean and does not consume Trainer's optional num_items_in_batch.
        # Transformers explicitly requires custom compute_loss implementations to set this false.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if "labels" not in inputs:
            raise ValueError("MLM training requires labels")
        labels = inputs["labels"]
        target_offsets = inputs.get("target_offsets")
        target_lengths = inputs.get("target_lengths")
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key != "labels" and key not in self.LOSS_METADATA_KEYS
        }
        outputs = model(**model_inputs)
        loss = calculate_mlm_loss(
            _logits_of(outputs),
            labels,
            config=self.loss_config,
            target_offsets=target_offsets,
            target_lengths=target_lengths,
        )
        return (loss, outputs) if return_outputs else loss
