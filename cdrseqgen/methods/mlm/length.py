"""Resolve fixed, bounded, and inferred CDR-length requests for the MLM method.

An encoder-only masked LM still needs one mask per emitted residue. Variable
length generation therefore has two phases: a length predictor sees one gap
sentinel, then the selected length is expanded into that many residue masks.
This module keeps the caller's constraints separate from the eventual sampled
length so benchmark artifacts say exactly which information was supplied.
"""

from dataclasses import dataclass
from typing import Optional


class UnsupportedLengthPolicy(ValueError):
    """Raised when a fixed-only caller receives a variable-length policy."""


def _validate_length(name: str, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class LengthConstraints:
    """Optional inclusive lower and upper bounds supplied by the caller."""

    minimum: Optional[int] = None
    maximum: Optional[int] = None

    def __post_init__(self):
        minimum = _validate_length("min_length", self.minimum)
        maximum = _validate_length("max_length", self.maximum)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(
                f"min_length ({minimum}) cannot exceed max_length ({maximum})"
            )

    @property
    def exact(self) -> Optional[int]:
        """Return the fixed length represented by equal bounds, if any."""
        if self.minimum is not None and self.minimum == self.maximum:
            return self.minimum
        return None

    @property
    def is_empty(self) -> bool:
        return self.minimum is None and self.maximum is None

    def contains(self, length: int) -> bool:
        """Whether ``length`` satisfies both inclusive bounds."""
        return (
            (self.minimum is None or length >= self.minimum)
            and (self.maximum is None or length <= self.maximum)
        )

    def as_dict(self) -> dict:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class LengthPolicy:
    """A caller request before a model-dependent length has been selected."""

    mode: str
    source: str
    constraints: LengthConstraints
    fixed: Optional[int] = None

    def __post_init__(self):
        if self.mode not in {"fixed", "bounded", "inferred"}:
            raise ValueError(f"unknown length-policy mode {self.mode!r}")
        if self.mode == "fixed":
            fixed = _validate_length("fixed", self.fixed)
            if self.constraints.exact != fixed:
                raise ValueError("a fixed policy must carry equal bounds")
        elif self.fixed is not None:
            raise ValueError("only a fixed policy may carry a fixed length")
        if self.mode == "bounded" and self.constraints.is_empty:
            raise ValueError("a bounded policy requires at least one bound")
        if self.mode == "inferred" and not self.constraints.is_empty:
            raise ValueError("an inferred policy cannot carry explicit bounds")

    @property
    def requires_predictor(self) -> bool:
        return self.fixed is None

    @property
    def length(self) -> int:
        """Compatibility accessor for callers that have already required fixed length."""
        if self.fixed is None:
            raise UnsupportedLengthPolicy(
                f"{self.mode} length policy requires a trained length predictor"
            )
        return self.fixed

    def as_dict(self, *, selected_lengths=None, probabilities=None) -> dict:
        out = {
            "mode": self.mode,
            "source": self.source,
            **self.constraints.as_dict(),
        }
        if self.fixed is not None:
            out["selected_length"] = self.fixed
        if selected_lengths is not None:
            selected = [int(length) for length in selected_lengths]
            out["selected_lengths"] = selected
            out["selected_length_counts"] = {
                str(length): selected.count(length) for length in sorted(set(selected))
            }
            if selected and len(set(selected)) == 1:
                out["selected_length"] = selected[0]
        if probabilities is not None:
            out["constrained_probabilities"] = {
                str(int(length)): float(probability)
                for length, probability in probabilities.items()
            }
        return out


@dataclass(frozen=True)
class LengthDecision:
    """A fixed length plus provenance describing how it was selected."""

    length: int
    source: str
    minimum: int
    maximum: int

    def __post_init__(self):
        _validate_length("length", self.length)
        _validate_length("minimum", self.minimum)
        _validate_length("maximum", self.maximum)
        if self.minimum != self.length or self.maximum != self.length:
            raise ValueError("a fixed-length decision must have equal bounds")

    def as_dict(self) -> dict:
        return {
            "mode": "fixed",
            "source": self.source,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "selected_length": self.length,
        }


def fixed_length(length: int, *, source: str) -> LengthDecision:
    """Create a validated fixed-length decision."""
    length = _validate_length("length", length)
    return LengthDecision(
        length=length,
        source=source,
        minimum=length,
        maximum=length,
    )


def native_mask_length(length: int) -> LengthDecision:
    """Describe the fixed-track eval/benchmark policy: preserve the native span."""
    return fixed_length(length, source="native_mask_count")


def fixed_policy(length: int, *, source: str) -> LengthPolicy:
    """Create the policy form of one exact length."""
    length = _validate_length("length", length)
    return LengthPolicy(
        mode="fixed",
        source=source,
        constraints=LengthConstraints(length, length),
        fixed=length,
    )


def resolve_length_policy(
    *,
    native_length: Optional[int] = None,
    use_native_length: bool = False,
    target_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> LengthPolicy:
    """Resolve user inputs without consulting the learned length predictor.

    ``native_length`` is deliberately ignored unless ``use_native_length`` is
    true. This prevents a native sequence retained for scoring from silently
    leaking its length into the length-free track.
    """
    native_length = _validate_length("native_length", native_length)
    target_length = _validate_length("target_length", target_length)
    constraints = LengthConstraints(minimum=min_length, maximum=max_length)

    if target_length is not None and not constraints.is_empty:
        raise ValueError(
            "target_length is a deprecated alias and cannot be combined with "
            "min_length or max_length"
        )

    requested = constraints.exact if constraints.exact is not None else target_length
    if use_native_length:
        if native_length is None:
            raise ValueError("use_native_length requires a native length")
        if requested is not None and requested != native_length:
            raise ValueError(
                f"requested length {requested} conflicts with native target length "
                f"{native_length}"
            )
        return fixed_policy(native_length, source="native_truth")

    if constraints.exact is not None:
        return fixed_policy(constraints.exact, source="explicit_bounds")
    if target_length is not None:
        return fixed_policy(target_length, source="legacy_target_length")
    if not constraints.is_empty:
        source = (
            "explicit_range" if constraints.minimum is not None and constraints.maximum is not None
            else "explicit_minimum" if constraints.minimum is not None
            else "explicit_maximum"
        )
        return LengthPolicy("bounded", source, constraints)
    return LengthPolicy("inferred", "length_predictor", constraints)


def resolve_generation_length_policy(
    mode: str,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
) -> LengthPolicy | None:
    """Validate an evaluation mode and return its non-native policy.

    ``None`` represents the legacy fixed-native track, whose mask count is read
    per example. Bounded and inferred tracks return one framework-independent
    request that is later resolved by the checkpoint's length predictor.
    """
    mode = str(mode).strip().lower()
    if mode not in {"fixed_native", "bounded", "inferred"}:
        raise ValueError("length mode must be fixed_native, bounded, or inferred")

    has_bounds = min_length is not None or max_length is not None
    if mode == "fixed_native":
        if has_bounds:
            raise ValueError("fixed_native mode does not accept min/max length bounds")
        return None
    if mode == "inferred":
        if has_bounds:
            raise ValueError("inferred mode does not accept min/max length bounds")
        return resolve_length_policy()

    if not has_bounds:
        raise ValueError("bounded mode requires at least one min/max length bound")
    if min_length is not None and min_length == max_length:
        raise ValueError("bounded mode does not accept equal bounds; use a fixed mode")
    policy = resolve_length_policy(min_length=min_length, max_length=max_length)
    if policy.mode != "bounded":
        raise ValueError("bounded mode requires non-equal min/max length bounds")
    return policy
