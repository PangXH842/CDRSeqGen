"""Shared backbone descriptors, ordered registries and loading helpers.

Workflow:
  method/backbones.py -> register(Backbone(...)) for each supported model family
  model_type -> method.resolve() -> Backbone
    -> build_tokenizer()       -- choose the vocabulary source and call its loader
    -> load_training_model()   -- build scratch weights or load a pretrained model
    -> load_pretrained_model() -- restore a checkpoint for generation/evaluation

Classes:
  * Backbone         -- tokenizer conventions and three method-supplied loaders
  * BackboneRegistry -- one ordered list per method; first matching predicate wins

Helpers:
  * tokenizer_source       -- prefer a saved tokenizer over the original model source
  * load_custom_pretrained -- rebuild a custom model and strictly restore its state dict
  * esm_tokenizer / bert_tokenizer -- shared protein tokenizer conventions

Each method owns its model loaders and registration blocks. AR extends Backbone's
tokenizer construction with its span-layout checks and persistence. Registries are
independent, so each method can register its own "scratch" backbone.
"""
import os
from dataclasses import dataclass
from typing import Callable

from cdrseqgen.datasets.dataset_common import TokenizerAdapter


@dataclass
class Backbone:
    """Describe one model family and the loaders used by the shared workflows.

    id / matches select the family. The three load_* callables construct its
    tokenizer, its training model and its restored checkpoint, respectively.
    position_type and uses_special_tokens pass input-layout conventions to the
    tokenizer loader; they do not configure the model's positional embeddings.
    fallback marks a generic entry, which must follow all named families.
    """

    id: str
    matches: Callable[[str], bool]
    position_type: str
    uses_special_tokens: bool
    load_tokenizer: Callable[[str, str, bool], TokenizerAdapter]
    load_training_model: Callable[..., object]
    load_pretrained_model: Callable[[str], object]
    fallback: bool = False

    def build_tokenizer(self, model_type, checkpoint_path=None) -> TokenizerAdapter:
        """Load the saved vocabulary when present, otherwise use model_type."""
        source = tokenizer_source(model_type, checkpoint_path)
        return self.load_tokenizer(source, self.position_type, self.uses_special_tokens)


class BackboneRegistry:
    """Store one method's entries in registration order.

    register() appends an entry; resolve() returns the first predicate match.
    Call validate() after registration: IDs must be unique; any fallback must be last.
    require_fallback=True requires exactly one; otherwise having none is valid.
    """

    def __init__(self, method, *, require_fallback=False, unsupported_hint=""):
        self.method = method
        self.require_fallback = require_fallback
        self.unsupported_hint = unsupported_hint
        self.entries: list[Backbone] = []

    def register(self, backbone: Backbone) -> Backbone:
        """Append and return a descriptor without loading its tokenizer or model."""
        self.entries.append(backbone)
        return backbone

    def resolve(self, model_type: str) -> Backbone:
        """Return the first matching entry or report the supported families."""
        for backbone in self.entries:
            if backbone.matches(model_type):
                return backbone
        hint = self.unsupported_hint or f"Add one in methods/{self.method}/backbones.py."
        raise ValueError(
            f"No {self.method} backbone registered for model_type={model_type!r}. "
            f"Registered backbones: {[b.id for b in self.entries]}. {hint}"
        )

    def validate(self):
        """Reject duplicate IDs and invalid generic-fallback counts or placement."""
        seen = set()
        for backbone in self.entries:
            if backbone.id in seen:
                raise ValueError(f"{self.method}: duplicate backbone id {backbone.id!r}")
            seen.add(backbone.id)

        fallbacks = [i for i, backbone in enumerate(self.entries) if backbone.fallback]
        if fallbacks == [len(self.entries) - 1]:
            return
        if not fallbacks and not self.require_fallback:
            return
        raise ValueError(
            f"{self.method}: expected {'one' if self.require_fallback else 'at most one'} "
            f"generic fallback, registered last; got fallbacks at {fallbacks} "
            f"of {[b.id for b in self.entries]}"
        )


def load_custom_pretrained(path, model_cls, config_cls):
    """Construct a custom model, strictly load model.safetensors and set evaluation mode."""
    from safetensors.torch import load_file
    model = model_cls(config_cls.from_pretrained(path))
    model.load_state_dict(load_file(os.path.join(path, 'model.safetensors')), strict=True)
    model.eval()
    return model


def esm_tokenizer(model_type, position_type, uses_special_tokens):
    """Adapt an ESM tokenizer using the supplied layout and wrapper conventions."""
    from transformers import EsmTokenizer
    return TokenizerAdapter.from_hf(
        EsmTokenizer.from_pretrained(model_type),
        position_type=position_type,
        uses_special_tokens=uses_special_tokens,
    )


def bert_tokenizer(model_type, position_type, uses_special_tokens):
    """Adapt a protein-BERT tokenizer with whitespace-separated residue encoding.

    Fall back to BertTokenizer for AutoTokenizer compatibility errors such as
    missing model metadata or optional dependencies. I/O errors propagate.
    """
    from transformers import AutoTokenizer, BertTokenizer
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_type, use_fast=False)
    except (ValueError, KeyError, ImportError):
        hf_tok = BertTokenizer.from_pretrained(model_type)
    return TokenizerAdapter.from_hf(
        hf_tok,
        position_type=position_type,
        uses_special_tokens=uses_special_tokens,
        space_residues=True,
    )


def tokenizer_source(model_type, checkpoint_path=None):
    """Prefer the exact saved vocabulary; retain support for older runs without tokenizer files."""
    if checkpoint_path and os.path.isfile(os.path.join(checkpoint_path, "tokenizer_config.json")):
        return str(checkpoint_path)
    return model_type
