"""Learn and apply the masked-LM gap-sentinel length classifier.

The sequence model remains an ordinary encoder-only masked LM. A small
checkpoint-specific classifier reads the final hidden state at one mask token,
predicts a distribution over lengths observed in the training split, and is
saved beside the sequence checkpoint. Generation then expands the sentinel to
the selected number of masks and reuses the existing residue decoder.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from cdrseqgen.artifacts import LENGTH_CONFIG_FILENAME, LENGTH_HEAD_FILENAME
from cdrseqgen.data.dataset import assemble_masked_input, framework_for_dataset, inject_context
from cdrseqgen.methods.mlm.length import LengthConstraints, LengthPolicy

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "cdrseqgen.length_predictor.v1"


@dataclass(frozen=True)
class GapPrompt:
    """One sentinel prompt and the position whose hidden state predicts length."""

    input_ids: list[int]
    gap_position: int
    reserve_length: int


def prepare_gap_prompt(
    tokenizer,
    pt1: str,
    pt2: str,
    *,
    partner_chain: str | None = None,
    antigen: str | None = None,
    max_len: int = 256,
    reserve_length: int,
) -> GapPrompt:
    """Build ``pt1 + [MASK] + pt2`` while reserving room for the longest class.

    Reserving a constant length makes context truncation independent of the
    unknown answer and identical between fitting and inference.
    """
    if reserve_length <= 0:
        raise ValueError("reserve_length must be positive")
    pt1_ids, pt2_ids = inject_context(
        tokenizer,
        tokenizer.encode(pt1),
        tokenizer.encode(pt2),
        reserve_length,
        partner_seq=partner_chain,
        antigen_seq=antigen,
        max_len=max_len,
    )
    input_ids, gap_position = assemble_masked_input(tokenizer, pt1_ids, pt2_ids, 1)
    return GapPrompt(input_ids, gap_position, reserve_length)


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
    """Fit a frozen-backbone length head and persist it beside the checkpoint."""
    from safetensors.torch import save_file

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

    torch.manual_seed(seed)
    head = LengthHead(train_x.size(1), len(supported_lengths)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    head.train()
    final_loss = None
    for _epoch in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for indices in _batched(order.tolist(), batch_size):
            x = train_x[indices].to(device)
            y = train_y[indices].to(device)
            optimizer.zero_grad()
            loss = loss_fn(head(x), y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())

    head.eval()
    with torch.no_grad():
        class_lengths = torch.tensor(supported_lengths, dtype=torch.long)
        train_predictions = class_lengths[head(train_x.to(device)).argmax(-1).cpu()]
        train_accuracy = float((train_predictions == train_lengths).float().mean())
        train_length_mae = float((train_predictions - train_lengths).abs().float().mean())
        if len(eval_x):
            eval_predictions = class_lengths[head(eval_x.to(device)).argmax(-1).cpu()]
            eval_accuracy = float((eval_predictions == eval_lengths).float().mean())
            eval_length_mae = float((eval_predictions - eval_lengths).abs().float().mean())
            eval_prediction_counts = Counter(int(length) for length in eval_predictions)
            if eval_supported.any():
                eval_supported_accuracy = float(
                    (eval_predictions[eval_supported] == eval_lengths[eval_supported])
                    .float()
                    .mean()
                )
            else:
                eval_supported_accuracy = None
        else:
            eval_accuracy = None
            eval_supported_accuracy = None
            eval_length_mae = None
            eval_prediction_counts = Counter()

    train_length_counts = Counter(int(length) for length in train_lengths)
    eval_length_counts = Counter(int(length) for length in eval_lengths)
    mode_length = min(
        supported_lengths,
        key=lambda length: (-train_length_counts[length], length),
    )
    if len(eval_lengths):
        mode_predictions = torch.full_like(eval_lengths, mode_length)
        eval_mode_accuracy = float((mode_predictions == eval_lengths).float().mean())
        eval_mode_length_mae = float(
            (mode_predictions - eval_lengths).abs().float().mean()
        )
    else:
        eval_mode_accuracy = None
        eval_mode_length_mae = None

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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in head.state_dict().items()},
        str(output_dir / LENGTH_HEAD_FILENAME),
    )
    with (output_dir / LENGTH_CONFIG_FILENAME).open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
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
