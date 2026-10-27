"""MLM length handling: policies, prompts, the learned length head, span sampling, evaluation strategies.

How a length request becomes residues:

    request                                       policy.mode  length comes from
    equal bounds, native truth or target_length   fixed        the request itself
    min and/or max length                         bounded      length head, renormalised inside the bounds
    nothing                                       inferred     length head over every length seen in training

    fixed              -> N masks -> residue decoder -> GeneratedSpan
    bounded / inferred -> gap-sentinel prompt -> head on the sentinel's hidden state -> one sampled
                          length per candidate -> candidates grouped by length -> masks -> decoder

Sections, in dependency order: 1 policies, 2 case requests, 3 prompts, 4 length head (inference,
then fitting), 5 span sampling, 6 evaluation strategies (Draw producers for eval.generative).

Callers:
  * training.finish_training / fit_mlm_length.py -> fit_length_predictor
  * generation callbacks -> resolve_case_length / LengthPredictor.load
  * case_generation / benchmark adapter -> sample_length_conditioned_spans
  * evaluation.run_generative_eval -> FixedNativeStrategy / VariableLengthStrategy

Fixed-native evaluation keeps the dataset's own mask count. Variable-length strategies see only
framework context and the fitted head; native truth stays in Draw for scoring. The head's checkpoint
filenames and schema (SCHEMA_VERSION) are unchanged. CLI setup, metrics, reporting and output
writing belong to the calling modules.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn

from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.datasets.dataset_mlm import (
    assemble_masked_input,
    framework_for_dataset,
    inject_context,
)
from cdrseqgen.eval.generative import Draw
from cdrseqgen.methods.mlm.sampling import pseudo_log_likelihood, sample_masked_span

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "cdrseqgen.length_predictor.v1"


# 1. Policies and constraints
#    What a request may say about length (bounds, an exact length) and the fixed | bounded |
#    inferred policy it resolves to. No model involved. Entry points: resolve_length_policy (one
#    case), resolve_generation_length_policy (a run-level mode string).


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
                str(length): count for length, count in sorted(Counter(selected).items())
            }
            if selected and len(set(selected)) == 1:
                out["selected_length"] = selected[0]
        if probabilities is not None:
            out["constrained_probabilities"] = {
                str(int(length)): float(probability)
                for length, probability in probabilities.items()
            }
        return out


def fixed_policy(length: int, *, source: str) -> LengthPolicy:
    """Create the policy form of one exact length."""
    length = _validate_length("length", length)
    return LengthPolicy("fixed", source, LengthConstraints(length, length), fixed=length)


def native_mask_length(length: int) -> LengthPolicy:
    """The fixed-track eval/benchmark policy: preserve the native span's mask count."""
    return fixed_policy(length, source="native_mask_count")


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
    return resolve_length_policy(min_length=min_length, max_length=max_length)


# 2. Case request resolution
#    A generation case dict plus CLI args -> (native truth, policy).


def resolve_case_length(case, args):
    """Return (normalized native truth, LengthPolicy) from case/CLI options.

    Equal bounds -> fixed; a range/one bound -> bounded; no bounds -> inferred.
    A present case key overrides args even when None. Native truth determines
    generation length only when use_native_length is explicitly enabled.
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


# 3. Prompt types and construction
#    Residue prompt: N masks for the decoder. Gap prompt: one sentinel mask, with capacity reserved
#    for the longest length class so truncation does not depend on the unknown native length.


@dataclass(frozen=True)
class FixedLengthPrompt:
    """A residue-mask prompt with context-side counts.

    *_truncated_tokens = bare flank count - final side count; values can be
    negative when optional context adds tokens. mask_positions marks the loop.
    """

    input_ids: list[int]
    mask_positions: list[int]
    target_length: int
    max_len: int
    pt1_tokens: int
    pt2_tokens: int
    pt1_truncated_tokens: int
    pt2_truncated_tokens: int

    def metadata(self) -> dict[str, int]:
        """Return stored prompt/side counts as a JSON-ready dict."""
        return {
            "max_len": self.max_len,
            "input_tokens": len(self.input_ids),
            "pt1_tokens": self.pt1_tokens,
            "pt2_tokens": self.pt2_tokens,
            "pt1_truncated_tokens": self.pt1_truncated_tokens,
            "pt2_truncated_tokens": self.pt2_truncated_tokens,
        }


def _prepare_masked_prompt(
    tokenizer, pt1, pt2, *, mask_length, reserve_length,
    partner_chain=None, antigen=None, max_len=256,
):
    """Assemble flanks/context with separate mask count and reserved capacity.

    Residue prompt: mask_length=N, reserve_length=N.
    Length prompt:  mask_length=1, reserve_length=longest supported class.
    """
    left, right = tokenizer.encode(pt1), tokenizer.encode(pt2)
    original_left, original_right = len(left), len(right)
    left, right = inject_context(
        tokenizer, left, right, reserve_length,
        partner_seq=partner_chain, antigen_seq=antigen, max_len=max_len,
    )
    input_ids, start = assemble_masked_input(tokenizer, left, right, mask_length)
    return FixedLengthPrompt(
        input_ids=input_ids, mask_positions=list(range(start, start + mask_length)),
        target_length=mask_length, max_len=max_len,
        pt1_tokens=len(left), pt2_tokens=len(right),
        pt1_truncated_tokens=original_left - len(left),
        pt2_truncated_tokens=original_right - len(right),
    )


def prepare_fixed_length_prompt(
    tokenizer, pt1: str, pt2: str, target_length: int, *,
    partner_chain: str | None = None, antigen: str | None = None, max_len: int = 256,
) -> FixedLengthPrompt:
    """Build pt1 + target_length residue masks + pt2, with context and wrappers.

    Example: pt1="AC", pt2="FG", target_length=3 -> AC [mask x 3] FG.
    The caller has already selected the length; this function validates it.
    """
    length = _validate_length("length", target_length)
    return _prepare_masked_prompt(
        tokenizer, pt1, pt2, mask_length=length, reserve_length=length,
        partner_chain=partner_chain, antigen=antigen, max_len=max_len,
    )


@dataclass(frozen=True)
class GapPrompt:
    """One sentinel prompt and the position whose hidden state predicts length."""

    input_ids: list[int]
    gap_position: int
    reserve_length: int


def prepare_gap_prompt(
    tokenizer, pt1: str, pt2: str, *, partner_chain: str | None = None,
    antigen: str | None = None, max_len: int = 256, reserve_length: int,
) -> GapPrompt:
    """Build one sentinel while reserving capacity for the longest length class.

    The constant reservation keeps fitting/inference truncation independent of
    the unknown native length. Only one actual mask enters the length head.
    """
    if reserve_length <= 0:
        raise ValueError("reserve_length must be positive")
    prompt = _prepare_masked_prompt(
        tokenizer, pt1, pt2, mask_length=1, reserve_length=reserve_length,
        partner_chain=partner_chain, antigen=antigen, max_len=max_len,
    )
    return GapPrompt(prompt.input_ids, prompt.mask_positions[0], reserve_length)


# 4. Learned length head: inference and fitting
#    A small classifier on the frozen model's hidden state at the gap sentinel, one class per length
#    seen in training. Inference: LengthPredictor.load / distribution / select. Fitting:
#    fit_length_predictor (features cached once, head trained, scored, saved beside the checkpoint).


def _hidden_states_of(outputs):
    if isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
    else:
        hidden_states = getattr(outputs, "hidden_states", None)
    if not hidden_states:
        raise ValueError(
            "backbone did not return hidden states; length prediction requires "
            "output_hidden_states=True support"
        )
    return hidden_states[-1]


@torch.no_grad()
def extract_gap_features(model, input_ids, attention_mask, gap_positions):
    """Return final-layer hidden vectors at each batch row's sentinel position."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = _hidden_states_of(outputs)
    rows = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, gap_positions]


class LengthHead(nn.Module):
    """Small classifier trained on frozen sequence-model sentinel features."""

    def __init__(self, feature_dim: int, n_lengths: int):
        super().__init__()
        self.normalization = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, n_lengths)

    def forward(self, features):
        return self.classifier(self.normalization(features))


class LengthPredictor:
    """Checkpoint artifact that selects fixed masks under optional bounds."""

    def __init__(self, head: LengthHead, config: dict):
        self.head = head
        self.config = config
        self.lengths = tuple(int(length) for length in config["supported_lengths"])
        self.reserve_length = int(config["reserve_length"])

    @classmethod
    def load(cls, checkpoint: str | os.PathLike, *, device=None) -> "LengthPredictor":
        from safetensors.torch import load_file

        checkpoint = Path(checkpoint)
        config_path = checkpoint / LENGTH_CONFIG_FILENAME
        weights_path = checkpoint / LENGTH_HEAD_FILENAME
        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"variable-length generation requires {LENGTH_CONFIG_FILENAME} and "
                f"{LENGTH_HEAD_FILENAME} in {checkpoint}; fit them with fit_mlm_length.py"
            )
        with config_path.open() as handle:
            config = json.load(handle)
        if config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported length predictor schema {config.get('schema_version')!r}")
        head = LengthHead(int(config["feature_dim"]), len(config["supported_lengths"]))
        head.load_state_dict(load_file(str(weights_path)), strict=True)
        if device is not None:
            head.to(device)
        head.eval()
        return cls(head, config)

    def _allowed(self, constraints: LengthConstraints) -> list[bool]:
        allowed = [constraints.contains(length) for length in self.lengths]
        if not any(allowed):
            observed = f"{min(self.lengths)}..{max(self.lengths)}"
            raise ValueError(
                f"length bounds {constraints.minimum}..{constraints.maximum} contain no "
                f"class learned by this checkpoint (observed {observed})"
            )
        return allowed

    @torch.no_grad()
    def distribution(
        self,
        model,
        tokenizer,
        device,
        *,
        pt1: str,
        pt2: str,
        policy: LengthPolicy,
        partner_chain: str | None = None,
        antigen: str | None = None,
        max_len: int = 256,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, GapPrompt]:
        """Return the constraint-renormalized length distribution."""
        if not policy.requires_predictor:
            raise ValueError("a fixed policy does not require the length predictor")
        if temperature < 0:
            raise ValueError("length temperature must be non-negative")
        prompt = prepare_gap_prompt(
            tokenizer,
            pt1,
            pt2,
            partner_chain=partner_chain,
            antigen=antigen,
            max_len=max_len,
            reserve_length=self.reserve_length,
        )
        ids = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        positions = torch.tensor([prompt.gap_position], dtype=torch.long, device=device)
        features = extract_gap_features(model, ids, attention, positions)
        self.head.to(device)
        logits = self.head(features)[0]
        allowed = torch.tensor(self._allowed(policy.constraints), dtype=torch.bool, device=device)
        logits = logits.masked_fill(~allowed, float("-inf"))
        scaled = logits if temperature == 0 else logits / temperature
        return torch.softmax(scaled, dim=-1), prompt

    def select(
        self,
        model,
        tokenizer,
        device,
        generator,
        *,
        pt1: str,
        pt2: str,
        policy: LengthPolicy,
        num_samples: int,
        partner_chain: str | None = None,
        antigen: str | None = None,
        max_len: int = 256,
        temperature: float = 1.0,
    ) -> tuple[list[int], dict[int, float], GapPrompt]:
        """Select one length per requested candidate, preserving stochastic draws."""
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        probabilities, prompt = self.distribution(
            model,
            tokenizer,
            device,
            pt1=pt1,
            pt2=pt2,
            policy=policy,
            partner_chain=partner_chain,
            antigen=antigen,
            max_len=max_len,
            temperature=temperature,
        )
        if temperature == 0:
            indices = [int(probabilities.argmax())] * num_samples
        else:
            indices = torch.multinomial(
                probabilities,
                num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        selected = [self.lengths[index] for index in indices]
        distribution = {
            length: float(probabilities[index].detach().cpu())
            for index, length in enumerate(self.lengths)
            if probabilities[index] > 0
        }
        return selected, distribution, prompt


def _batched(values: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def _length_scores(predicted, native):
    """Return exact-length accuracy and MAE, or (None, None) for an empty set."""
    if not len(native):
        return None, None
    return (
        float((predicted == native).float().mean()),
        float((predicted - native).abs().float().mean()),
    )


@torch.no_grad()
def _extract_dataset_features(
    model,
    tokenizer,
    dataset,
    *,
    device,
    batch_size: int,
    reserve_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = []
    labels = []
    for index in range(len(dataset)):
        framework = framework_for_dataset(dataset, index)
        prompts.append(prepare_gap_prompt(
            tokenizer,
            framework["pt1"],
            framework["pt2"],
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
            reserve_length=reserve_length,
        ))
        labels.append(int(framework["target_length"]))

    if not prompts:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    features = []
    for prompt_batch in _batched(prompts, batch_size):
        tensors = [torch.tensor(prompt.input_ids, dtype=torch.long) for prompt in prompt_batch]
        ids = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=tokenizer.pad_id
        ).to(device)
        attention = (ids != tokenizer.pad_id).long()
        positions = torch.tensor(
            [prompt.gap_position for prompt in prompt_batch], dtype=torch.long, device=device
        )
        features.append(extract_gap_features(model, ids, attention, positions).cpu())
    return torch.cat(features), torch.tensor(labels, dtype=torch.long)


def _train_head(train_x, train_y, *, n_classes, epochs, learning_rate, batch_size, seed, device):
    """Fit the classifier on cached features; return (head in eval mode, last batch loss)."""
    torch.manual_seed(seed)
    head = LengthHead(train_x.size(1), n_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    head.train()
    final_loss = None
    for _epoch in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for indices in _batched(order.tolist(), batch_size):
            optimizer.zero_grad()
            loss = loss_fn(head(train_x[indices].to(device)), train_y[indices].to(device))
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    head.eval()
    return head, final_loss


@torch.no_grad()
def _predict_lengths(head, features, class_lengths, device):
    """Argmax class of every cached feature row, mapped back to its length."""
    return class_lengths[head(features.to(device)).argmax(-1).cpu()]


def _save_predictor(output_dir, head, config):
    """Write the head weights and the JSON config beside the sequence-model checkpoint."""
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
        str(output_dir / LENGTH_HEAD_FILENAME),
    )
    with (output_dir / LENGTH_CONFIG_FILENAME).open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def fit_length_predictor(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    *,
    output_dir: str | os.PathLike,
    device,
    epochs: int = 20,
    learning_rate: float = 1e-2,
    batch_size: int = 256,
    seed: int = 42,
) -> LengthPredictor:
    """Fit a frozen-backbone length head and persist it beside the checkpoint.

    Phases: cache sentinel features -> train the classifier -> score it against the train/eval
    lengths and a train-mode baseline -> write the weights and a JSON config of everything above.
    """
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("length predictor epochs, batch size, and learning rate must be positive")
    if not len(train_dataset):
        raise ValueError("cannot fit a length predictor on an empty training dataset")

    supported_lengths = sorted({
        int(framework_for_dataset(train_dataset, index)["target_length"])
        for index in range(len(train_dataset))
    })
    reserve_length = max(supported_lengths)
    was_training = model.training
    model.eval()
    logger.info(
        "📏 Extracting gap-sentinel features for %d train / %d eval frameworks...",
        len(train_dataset), len(eval_dataset),
    )
    train_x, train_lengths = _extract_dataset_features(
        model, tokenizer, train_dataset, device=device, batch_size=batch_size,
        reserve_length=reserve_length,
    )
    eval_x, eval_lengths = _extract_dataset_features(
        model, tokenizer, eval_dataset, device=device, batch_size=batch_size,
        reserve_length=reserve_length,
    )
    if was_training:
        model.train()

    class_of = {length: index for index, length in enumerate(supported_lengths)}
    train_y = torch.tensor([class_of[int(length)] for length in train_lengths])
    eval_supported = torch.tensor([int(length) in class_of for length in eval_lengths])

    head, final_loss = _train_head(
        train_x, train_y, n_classes=len(supported_lengths), epochs=epochs,
        learning_rate=learning_rate, batch_size=batch_size, seed=seed, device=device,
    )

    class_lengths = torch.tensor(supported_lengths, dtype=torch.long)
    train_predictions = _predict_lengths(head, train_x, class_lengths, device)
    train_accuracy, train_length_mae = _length_scores(train_predictions, train_lengths)
    if len(eval_x):
        eval_predictions = _predict_lengths(head, eval_x, class_lengths, device)
        eval_accuracy, eval_length_mae = _length_scores(eval_predictions, eval_lengths)
        eval_prediction_counts = Counter(int(length) for length in eval_predictions)
        eval_supported_accuracy = (
            float((eval_predictions[eval_supported] == eval_lengths[eval_supported]).float().mean())
            if eval_supported.any() else None
        )
    else:
        eval_accuracy = eval_supported_accuracy = eval_length_mae = None
        eval_prediction_counts = Counter()

    train_length_counts = Counter(int(length) for length in train_lengths)
    eval_length_counts = Counter(int(length) for length in eval_lengths)
    mode_length = min(supported_lengths, key=lambda length: (-train_length_counts[length], length))
    # Always predicting the train-mode length is the baseline the head has to beat. With no eval
    # frameworks _length_scores returns (None, None).
    eval_mode_accuracy, eval_mode_length_mae = _length_scores(
        torch.full_like(eval_lengths, mode_length), eval_lengths)

    config = {
        "schema_version": SCHEMA_VERSION,
        "feature_type": "last_hidden_state_at_gap_sentinel",
        "feature_dim": int(train_x.size(1)),
        "supported_lengths": supported_lengths,
        "reserve_length": reserve_length,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "eval_supported_examples": int(eval_supported.sum()),
        "eval_unseen_lengths": sorted(set(eval_length_counts) - set(supported_lengths)),
        "train_length_counts": {
            str(length): train_length_counts[length] for length in supported_lengths
        },
        "eval_length_counts": {
            str(length): eval_length_counts[length] for length in sorted(eval_length_counts)
        },
        "eval_prediction_counts": {
            str(length): eval_prediction_counts[length] for length in sorted(eval_prediction_counts)
        },
        "train_mode_length": mode_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "final_batch_loss": final_loss,
        "train_accuracy": train_accuracy,
        "train_length_mae": train_length_mae,
        "eval_accuracy": eval_accuracy,
        "eval_supported_accuracy": eval_supported_accuracy,
        "eval_length_mae": eval_length_mae,
        "eval_mode_baseline_accuracy": eval_mode_accuracy,
        "eval_mode_baseline_length_mae": eval_mode_length_mae,
    }
    _save_predictor(output_dir, head, config)
    logger.info(
        "📏 Length predictor saved: train accuracy %.3f, eval accuracy %s "
        "(mode baseline %s), eval MAE %s, classes=%s",
        train_accuracy,
        f"{eval_accuracy:.3f}" if eval_accuracy is not None else "n/a",
        f"{eval_mode_accuracy:.3f}" if eval_mode_accuracy is not None else "n/a",
        f"{eval_length_mae:.3f}" if eval_length_mae is not None else "n/a",
        supported_lengths,
    )
    return LengthPredictor(head, config)


# 5. Length-conditioned residue sampling
#    policy -> one length per candidate -> one prompt and decoder call per distinct length. The
#    single entry point shared by generation (case_generation), evaluation and the benchmark.


@dataclass(frozen=True)
class GeneratedSpan:
    """One candidate's token IDs, selected length and scoring prompt.

    input_ids/mask_positions preserve its exact context for later self-scoring.
    """

    token_ids: list[int]
    length: int
    input_ids: list[int]
    mask_positions: list[int]
    prompt_metadata: dict[str, int]


def sample_length_conditioned_spans(
    model, tokenizer, device, generator, *, pt1: str, pt2: str, policy: LengthPolicy,
    length_predictor=None, partner_chain: str | None = None, antigen: str | None = None,
    max_len: int = 256, decoding: str, num_samples: int, temperature: float,
    top_k: int, top_p: float, length_temperature: float = 1.0,
) -> tuple[list[GeneratedSpan], dict]:
    """Select lengths, decode grouped prompts and return ordered spans plus metadata.

    Example: selected lengths [8, 10, 8] -> groups {8: [0, 2], 10: [1]}.
    Returned spans retain order [8, 10, 8] and their individual scoring prompts.

    Fixed policies repeat their selected length; bounded/inferred policies need
    the saved head. Residue temperature=0 gives one draw. length_temperature
    independently controls length sampling. No native truth enters this API.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    count = 1 if temperature == 0 else num_samples
    context = dict(
        pt1=pt1, pt2=pt2, partner_chain=partner_chain, antigen=antigen, max_len=max_len,
    )
    probabilities = gap_prompt = None
    if policy.requires_predictor:
        if length_predictor is None:
            raise FileNotFoundError(
                "bounded/inferred generation requires this checkpoint's trained length predictor"
            )
        lengths, probabilities, gap_prompt = length_predictor.select(
            model, tokenizer, device, generator, policy=policy, num_samples=count,
            temperature=length_temperature, **context,
        )
    else:
        lengths = [policy.length] * count

    groups = defaultdict(list)
    for index, length in enumerate(lengths):
        groups[length].append(index)
    results, prompts = [None] * count, {}
    for length, positions in groups.items():
        prompt = prepare_fixed_length_prompt(tokenizer, target_length=length, **context)
        prompts[str(length)] = prompt.metadata()
        sampled = sample_masked_span(
            model, tokenizer, device, generator, prompt.input_ids, prompt.mask_positions,
            decoding=decoding, num_samples=len(positions), temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
        for index, ids in zip(positions, sampled):
            results[index] = GeneratedSpan(
                token_ids=ids, length=length, input_ids=prompt.input_ids,
                mask_positions=prompt.mask_positions, prompt_metadata=prompt.metadata(),
            )

    metadata = policy.as_dict(selected_lengths=lengths, probabilities=probabilities)
    metadata.update(length_temperature=length_temperature, prompts_by_length=prompts)
    if gap_prompt is not None:
        metadata["gap_prompt"] = dict(
            input_tokens=len(gap_prompt.input_ids),
            gap_position=gap_prompt.gap_position,
            reserve_length=gap_prompt.reserve_length,
        )
    return results, metadata


# 6. Fixed-native and variable-length evaluation adapters
#    Draw producers for eval.generative: fixed-native fills the dataset's own masks, variable-length
#    uses the policy and the head. Both expose draw / self_scores / judge_inputs / payload_length_policy.


def masked_view(item):
    """Return prompt IDs, supervised positions and native IDs.

    Example: labels [-100, 3, 4, -100] -> positions [1, 2], native [3, 4].
    Positions follow labels; fixed-native evaluation expects fully masked inputs.
    """
    input_ids = item["input_ids"].tolist()
    labels = item["labels"].tolist()
    mask_positions = [p for p, l in enumerate(labels) if l != -100]
    native_ids = [labels[p] for p in mask_positions]
    return input_ids, mask_positions, native_ids


class _EvaluationStrategy:
    """Shared MLM evaluation state, candidate self-scoring and Subset index mapping."""

    def __init__(self, *, model, tokenizer, device, generator, eval_dataset, judge_dataset,
                 judge_tokenizer, decoding, num_samples, temperature, top_k, top_p):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.generator = generator
        self.eval_dataset, self.judge_dataset = eval_dataset, judge_dataset
        self.judge_tokenizer = judge_tokenizer
        self.decoding, self.num_samples = decoding, num_samples
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self._judge_cache = (None, None)

    def self_scores(self, drawn):
        """Re-mask each candidate position and return pseudo-perplexities in draw order."""
        return [
            pseudo_log_likelihood(
                self.model, ids, positions, tokens, self.tokenizer.special_ids,
                self.tokenizer.mask_id, self.device,
            )[1]
            for ids, positions, tokens in drawn.spans
        ]

    def _full_index(self, i):
        return self.eval_dataset.indices[i] if hasattr(self.eval_dataset, "indices") else i

    def _cached_judge(self, i, build):
        """One-entry cache for the judge side of framework i; build(full_index) fills it.

        The judge dataset uses the full corpus order, so the eval Subset index is mapped first.
        """
        if self._judge_cache[0] != i:
            self._judge_cache = (i, build(self._full_index(i)))
        return self._judge_cache[1]


class FixedNativeStrategy(_EvaluationStrategy):
    """Fill the dataset's masks using the native loop length.

    Generator context comes from eval_dataset; judge prompts use judge_dataset.
    """

    supplies_native_length = True

    def payload_length_policy(self):
        """Describe the native-mask-count policy for the evaluation payload."""
        return {"mode": "fixed", "source": "native_mask_count"}

    def draw(self, i):
        """Return candidates and scoring prompts for index i, or None if no masks exist."""
        input_ids, mask_positions, native_ids = masked_view(self.eval_dataset[i])
        if not mask_positions:
            return None
        policy = native_mask_length(len(mask_positions))
        native = self.tokenizer.decode(native_ids).upper().strip()
        sampled_ids = sample_masked_span(
            self.model, self.tokenizer, self.device, self.generator,
            input_ids, mask_positions,
            decoding=self.decoding, num_samples=self.num_samples,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
        )
        spans = [(input_ids, mask_positions, ids) for ids in sampled_ids]
        seqs = [self.tokenizer.decode(ids).upper().strip() for ids in sampled_ids]
        return Draw(
            native=native,
            seqs=seqs,
            spans=spans,
            target_len=policy.length,
            length_policy=policy.as_dict(),
            record_extra={},
        )

    def judge_inputs(self, i, _seq):
        """The judge's native-length masked prompt for framework i; the candidate does not change it."""
        return self._cached_judge(i, lambda full: masked_view(self.judge_dataset[full])[:2])


class VariableLengthStrategy(_EvaluationStrategy):
    """Select lengths from the fitted head before filling residue masks.

    Each candidate retains its own prompt; the native loop is scoring-only.
    """

    supplies_native_length = False

    def __init__(self, *, length_predictor, length_policy, length_temperature, **shared_options):
        super().__init__(**shared_options)
        self.length_predictor = length_predictor
        self.length_policy = length_policy
        self.length_temperature = length_temperature

    def payload_length_policy(self):
        """Return the requested policy and length-sampling temperature for the payload."""
        return {**self.length_policy.as_dict(), "length_temperature": self.length_temperature}

    def draw(self, i):
        """Generate from framework context and retain native truth only in the Draw."""
        framework = framework_for_dataset(self.eval_dataset, i)
        native = framework["target"].upper().strip()
        generated, length_metadata = sample_length_conditioned_spans(
            self.model,
            self.tokenizer,
            self.device,
            self.generator,
            pt1=framework["pt1"],
            pt2=framework["pt2"],
            policy=self.length_policy,
            length_predictor=self.length_predictor,
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
            decoding=self.decoding,
            num_samples=self.num_samples,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            length_temperature=self.length_temperature,
        )
        seqs = [self.tokenizer.decode(span.token_ids).upper().strip() for span in generated]
        return Draw(
            native=native,
            seqs=seqs,
            spans=[(span.input_ids, span.mask_positions, span.token_ids) for span in generated],
            # The native length labels and buckets the example; generation never received it.
            target_len=len(native),
            length_policy=length_metadata,
            record_extra={"generated_lengths": [len(sequence) for sequence in seqs]},
        )

    def judge_inputs(self, i, seq):
        """Build a judge prompt with len(seq) masks; the framework is cached per index.

        Example: native length=8, candidate length=6 -> six judge masks.
        """
        framework = self._cached_judge(
            i, lambda full: framework_for_dataset(self.judge_dataset, full))
        prompt = prepare_fixed_length_prompt(
            self.judge_tokenizer,
            framework["pt1"],
            framework["pt2"],
            len(seq),
            partner_chain=framework.get("partner_chain"),
            antigen=framework.get("antigen"),
            max_len=framework["max_len"],
        )
        return prompt.input_ids, prompt.mask_positions
