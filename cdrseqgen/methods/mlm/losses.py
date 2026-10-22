"""Define and register training losses for masked-LM CDR reconstruction.

Workflow:
  config/CLI -> loss_config_from_args() -> LossConfig
  logits + labels + optional target metadata
    -> calculate_mlm_loss() -> registered loss function -> scalar loss

Loss blocks, in registration order:
  * cross_entropy -- mean CE over supervised tokens
  * weighted_cross_entropy -- CE weighted by original loop core/anchor positions

Classes:
  * LossConfig -- normalized, validated settings for the selected loss
  * MLMLossTrainer -- connect the shared Trainer workflow to these calculations

Extending:
  * Add a calculation and its register_loss(...) call together in a new block.
  * Calculations accept (logits, labels, target_offsets, target_lengths, config)
    and return a scalar tensor. Declare aliases and metadata needs at registration.

Shared token CE lives in common.losses; the model-forward/loss handoff lives
in common.training. Core/anchor weighting and loss selection stay here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from cdrseqgen.methods.common.losses import (
    IGNORE_INDEX,
    token_cross_entropy as _token_cross_entropy,
)
from cdrseqgen.methods.common.training import ExternalLossTrainer


# 1. Registry: each loss block below supplies its calculation and options.
LOSS_REGISTRY: dict[str, Callable] = {}
LOSS_TYPES: tuple[str, ...] = ()
_LOSS_ALIASES: dict[str, str] = {}
_LOSSES_REQUIRING_METADATA: set[str] = set()


def register_loss(name, calculation, *, aliases=(), requires_target_metadata=False):
    """Register a calculation under a canonical name and optional aliases.

    Example:
      register_loss("cross_entropy", _cross_entropy, aliases=("ce",))
      -> both names select _cross_entropy; LOSS_TYPES lists only "cross_entropy"

    Names use lowercase words separated by underscores. Duplicate names and
    aliases are rejected. LOSS_REGISTRY values remain directly callable.
    """
    global LOSS_TYPES
    names = (name, *aliases)
    if len(set(names)) != len(names) or any(
        item in LOSS_REGISTRY or item in _LOSS_ALIASES for item in names
    ):
        raise ValueError(f"loss name or alias already registered: {names!r}")
    LOSS_REGISTRY[name] = calculation
    _LOSS_ALIASES.update({alias: name for alias in aliases})
    if requires_target_metadata:
        _LOSSES_REQUIRING_METADATA.add(name)
    LOSS_TYPES = tuple(LOSS_REGISTRY)
    return calculation


def normalize_loss_type(loss_type: str) -> str:
    """Return a canonical loss name or raise a configuration error."""
    normalized = str(loss_type).strip().lower().replace("-", "_")
    normalized = _LOSS_ALIASES.get(normalized, normalized)
    if normalized not in LOSS_REGISTRY:
        raise ValueError(f"loss_type must be one of {LOSS_TYPES}, got {loss_type!r}")
    return normalized


# 2. Configuration: keep the existing config fields and argument conversion.
@dataclass(frozen=True)
class LossConfig:
    """Store validated settings for one registered MLM loss.

    Example:
      LossConfig("weighted", core_weight=3, anchor_weight=1, anchor_trim=2)
      -> weighted_cross_entropy; core tokens receive three times the anchor weight

    Plain CE ignores the core/anchor settings. Both objectives support label
    smoothing. requires_target_metadata comes from the selected registration.
    """

    loss_type: str = "cross_entropy"
    core_weight: float = 1.0
    anchor_weight: float = 1.0
    anchor_trim: int = 2
    label_smoothing: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "loss_type", normalize_loss_type(self.loss_type))
        weights = (self.core_weight, self.anchor_weight)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("core_loss_weight and anchor_loss_weight must be finite and non-negative")
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
        return self.loss_type in _LOSSES_REQUIRING_METADATA


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


# 3. Cross-entropy: one calculation followed by its registration.
def _cross_entropy(logits, labels, _target_offsets, _target_lengths, config):
    """Return sum(token CE) / number of supervised tokens in this microbatch."""
    return _token_cross_entropy(
        logits, labels, label_smoothing=config.label_smoothing, reduction="mean"
    )


register_loss("cross_entropy", _cross_entropy, aliases=("ce", "crossentropy"))


# 4. Weighted cross-entropy: classify positions, calculate, then register.
def _core_anchor_weights(logits, labels, target_offsets, target_lengths, config):
    """Return one weight per sequence position, with zero at ignored labels.

    Example: length=8, anchor_trim=2, core_weight=3, anchor_weight=1
      loop offset: 0 1 2 3 4 5 6 7
      region:      A A C C C C A A
      full mask:   1 1 3 3 3 3 1 1
      mixed mask:  0 1 3 0 0 3 1 0   (only selected offsets are supervised)

    Classify by the original loop offset, including under mixed masking.
    If length <= 2 * anchor_trim, all supervised positions are anchors.
    """
    if target_offsets.shape != labels.shape:
        raise ValueError(
            f"target_offsets shape {tuple(target_offsets.shape)} does not match labels "
            f"{tuple(labels.shape)}"
        )
    if target_lengths.ndim != 1 or target_lengths.size(0) != labels.size(0):
        raise ValueError("target_lengths must contain one length per batch row")

    active = labels.ne(IGNORE_INDEX)
    valid_offsets = target_offsets[active]
    active_lengths = target_lengths[:, None].expand_as(labels)[active]
    if bool(((valid_offsets < 0) | (valid_offsets >= active_lengths)).any()):
        raise ValueError("supervised labels carry invalid target offsets")

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
    return token_weights * active.to(logits.dtype)


def _weighted_cross_entropy(logits, labels, target_offsets, target_lengths, config):
    """Return sum(weight * token CE) / sum(weights) over this microbatch.

    Example: four anchor weights of 1 and four core weights of 3
      denominator = 4*1 + 4*3 = 16

    Unit weights reproduce plain CE. Zero weights exclude a region; a batch
    with no positive supervised weight raises ValueError. Smoothing is applied
    to each token's CE before weighting.
    """
    token_weights = _core_anchor_weights(logits, labels, target_offsets, target_lengths, config)

    token_losses = _token_cross_entropy(
        logits, labels, label_smoothing=config.label_smoothing, reduction="none"
    ).reshape_as(labels)
    total_weight = token_weights.sum()
    if not bool(total_weight > 0):
        raise ValueError(
            f"every supervised token in this batch has weight zero (core={config.core_weight}, "
            f"anchor={config.anchor_weight}, trim={config.anchor_trim}). "
            "Adjust the weights or masking/anchor_trim so at least one supervised token "
            "has positive weight."
        )
    return (token_losses * token_weights).sum() / total_weight


register_loss(
    "weighted_cross_entropy",
    _weighted_cross_entropy,
    aliases=("weighted", "weighted_ce"),
    requires_target_metadata=True,
)


# 5. Public calculation entry: validate the batch and dispatch to its loss.
def calculate_mlm_loss(
    logits,
    labels,
    *,
    config: LossConfig,
    target_offsets=None,
    target_lengths=None,
):
    """Validate aligned MLM tensors and return the configured scalar loss.

    Inputs:
      logits:         [B, S, V] predictions at the labelled positions (no shift)
      labels:         [B, S] token IDs; -100 means unsupervised
      target_offsets: [B, S] original loop offsets, required by weighted CE
      target_lengths: [B] original loop lengths, required by weighted CE

    Reject malformed shapes, entirely unsupervised batches and missing metadata
    before invoking the registered calculation. Reduction is per microbatch,
    preserving the existing Trainer gradient-accumulation behavior.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(
            f"expected logits [B,S,V] and labels [B,S], got {tuple(logits.shape)} and "
            f"{tuple(labels.shape)}"
        )
    if not bool(labels.ne(IGNORE_INDEX).any()):
        raise ValueError("cannot calculate MLM loss for a batch with no supervised tokens")
    if config.requires_target_metadata and (target_offsets is None or target_lengths is None):
        raise ValueError(
            f"{config.loss_type} requires target_offsets and target_lengths from mlm_collate_fn"
        )
    return LOSS_REGISTRY[config.loss_type](
        logits, labels, target_offsets, target_lengths, config
    )


# 6. MLM Trainer adapter: retain the existing import and constructor interface.
def _logits_of(outputs):
    """Extract logits from dict, ModelOutput-like or tuple model outputs."""
    if isinstance(outputs, dict):
        return outputs["logits"]
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[0]


class MLMLossTrainer(ExternalLossTrainer):
    """Apply the configured MLM objective through the shared Trainer workflow.

    ExternalLossTrainer owns compute_loss(): separate inputs, forward the
    model, and return loss/outputs. This subclass supplies the MLM settings,
    metadata keys and loss_from_outputs() calculation.
    """

    LOSS_METADATA_KEYS = {"target_offsets", "target_lengths"}

    def __init__(self, *args, loss_config: LossConfig | None = None, **kwargs):
        self.loss_config = loss_config or LossConfig()
        super().__init__(*args, **kwargs)

    def loss_from_outputs(self, outputs, labels, inputs):
        return calculate_mlm_loss(
            _logits_of(outputs),
            labels,
            config=self.loss_config,
            target_offsets=inputs.get("target_offsets"),
            target_lengths=inputs.get("target_lengths"),
        )
