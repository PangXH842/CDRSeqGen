"""Causal backbone registry — resolve(model_type) -> CausalBackbone.

Deliberately the same shape as ``methods/mlm/backbones.py``: one entry per model family, a
``matches`` predicate, a generic fallback registered LAST, and an assertion that keeps it there.
Adding a causal model is one ``register(CausalBackbone(...))`` entry; entrypoints never branch on
model type.

  * scratch     -- the from-scratch causal decoder (models/transformer.py)
  * hf_decoder  -- generic AutoModelForCausalLM fallback: any hub id or local dir

**The pretrained case carries a real cost, and it is not hidden here.** The span-infilling layout
needs three sentinels — a gap marker, a prompt/answer separator, and an answer terminator. A
from-scratch causal vocab has all three (``dataset_ar.CausalTokenizer``). A pretrained protein LM
such as ProtGPT2 or ProGen2 has an EOS and nothing else: it was trained on plain left-to-right
sequences and has never seen a span-infilling format. So ``hf_decoder`` *adds* the missing sentinels
and resizes the embedding matrix, which means those rows start random and the model must learn a
format it was not pretrained for.

That is worth stating plainly because it bounds what Phase D can conclude: IgLM's advantage in
Findings §11 is not merely "autoregressive", it is "autoregressive **and** pretrained natively on
antibody span infilling". A fine-tuned ProtGPT2 is a weaker claim than a matched one, and the
pretraining gain should be expected to fall short of the +0.12-0.24 that §1 measured for ESM-2.
"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from cdrseqgen.datasets.dataset_ar import (CausalTokenizer, causal_adapter, CDRSEQGEN_CAUSAL,
                                         layout_for, validate_layout, span_layout_metadata)
from cdrseqgen.artifacts import atomic_write_json, find_run_config
from cdrseqgen.datasets.dataset_common import TokenizerAdapter
from cdrseqgen.methods.common.backbones import (
    load_custom_pretrained as _load_custom_pretrained, tokenizer_source)

logger = logging.getLogger(__name__)

LAYOUT_FILENAME = "span_layout.json"


def validate_checkpoint_layout(path, tokenizer):
    """All recorded contracts must agree with the live tokenizer/layout; absent legacy is explicit."""
    expected = span_layout_metadata(tokenizer)
    root = Path(path)
    recorded = []
    sidecar = root / LAYOUT_FILENAME
    if sidecar.is_file():
        with sidecar.open() as handle:
            recorded.append((sidecar, json.load(handle)))
    for source in (root / 'config.json', find_run_config(root)):
        if source is not None and Path(source).is_file():
            with Path(source).open() as handle:
                payload = json.load(handle)
            if 'span_layout' in payload:
                recorded.append((source, payload['span_layout']))
    for source, actual in recorded:
        if actual != expected or any(type(actual[key]) is not type(value)
                                     for key, value in expected.items()):
            raise ValueError(f"Span layout mismatch in {source}: recorded {actual!r}, "
                             f"requested {expected!r}. Layout reinterpretation is not supported.")
    if not recorded:
        # Historical checkpoints used this precise v1 format. Keep this fixed even if the
        # current default changes: missing provenance must never opt an old model into v2.
        legacy = {"id": "cdrseqgen_causal", "version": 1, "n_layout_tokens": 3,
                  "terminator_id": tokenizer.eos_id}
        if expected != legacy:
            raise ValueError("Checkpoint has no span layout metadata; only the legacy default is supported")
        logger.warning("Legacy checkpoint %s has no span layout metadata; assuming %s version %s",
                       path, expected['id'], expected['version'])
    return expected


def persist_layout_with_tokenizer(tokenizer):
    """Save the contract alongside native vocabularies, including each Trainer checkpoint."""
    save_native = getattr(tokenizer, 'save_pretrained', None)

    def save(path):
        metadata = span_layout_metadata(tokenizer)
        Path(path).mkdir(parents=True, exist_ok=True)
        if callable(save_native):
            save_native(path)
        atomic_write_json(Path(path) / LAYOUT_FILENAME, metadata)

    tokenizer.save_pretrained = save
    return tokenizer


def record_training_layout(model, tokenizer, output_dir):
    """Embed the contract in model config and run provenance before training/save callbacks."""
    metadata = span_layout_metadata(tokenizer)
    model.config.span_layout = metadata
    source = find_run_config(output_dir)
    if source is None:
        raise ValueError("Training layout requires an initialized run_config.json")
    with Path(source).open() as handle:
        run_config = json.load(handle)
    run_config['span_layout'] = metadata
    atomic_write_json(source, run_config)


# The sentinels the causal layout needs beyond a plain LM vocabulary.
REQUIRED_SENTINELS = ("<mask>", "<sep>", "<eos>")


@dataclass
class CausalBackbone:
    id: str
    matches: Callable[[str], bool]
    position_type: str          # "relative" for rotary/relative models, "absolute" otherwise
    uses_special_tokens: bool
    load_tokenizer: Callable[[str, str, bool], TokenizerAdapter]
    load_training_model: Callable[..., object]
    load_pretrained_model: Callable[[str], object]
    fallback: bool = False

    def build_tokenizer(self, model_type, checkpoint_path=None) -> TokenizerAdapter:
        source = tokenizer_source(model_type, checkpoint_path)
        tokenizer = self.load_tokenizer(source, self.position_type, self.uses_special_tokens)
        validate_layout(layout_for(tokenizer), tokenizer)
        if checkpoint_path:
            validate_checkpoint_layout(checkpoint_path, tokenizer)
        return persist_layout_with_tokenizer(tokenizer)


REGISTRY = []


def register(backbone: CausalBackbone) -> CausalBackbone:
    REGISTRY.append(backbone)
    return backbone


def resolve(model_type: str) -> CausalBackbone:
    """Return the first causal backbone whose predicate matches, else a helpful error."""
    for backbone in REGISTRY:
        if backbone.matches(model_type):
            return backbone
    raise ValueError(
        f"No causal backbone registered for model_type={model_type!r}. "
        f"Registered backbones: {[b.id for b in REGISTRY]}. Add one in methods/ar/backbones.py."
    )


# --- custom from-scratch causal decoder ---------------------------------------------------------
def _scratch():
    from cdrseqgen.methods.ar.models import transformer
    return transformer


def _build_scratch(mt, vocab_size=None, hidden_size=128, num_layers=4, num_attention_heads=8,
                   intermediate_size=None, max_len=512, **_ignored):
    # Default to the causal vocab (27) rather than the masked one (25): the two extra rows are
    # <bos>/<eos>, and sizing to 25 would make <eos> unrepresentable.
    if vocab_size is None:
        vocab_size = len(CausalTokenizer().vocab)
    return _scratch().build_model(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
        num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
        max_len=max_len)


def _load_scratch_pretrained(path):
    validate_checkpoint_layout(path, causal_adapter(position_type="relative"))
    t = _scratch()
    return _load_custom_pretrained(path, t.CausalAntibodyModel, t.CausalAntibodyConfig)


register(CausalBackbone(
    id="scratch",
    matches=lambda mt: mt == "scratch",
    position_type="relative",       # rotary — see models/transformer.py and Findings §8
    uses_special_tokens=False,
    load_tokenizer=lambda mt, pos, wrap: causal_adapter(position_type=pos),
    load_training_model=_build_scratch,
    load_pretrained_model=_load_scratch_pretrained,
))


# --- generic Hugging Face causal LM --------------------------------------------------------------
def missing_sentinels(hf_tokenizer):
    """Which of the layout's sentinels this tokenizer lacks, in registry order."""
    vocab = hf_tokenizer.get_vocab()
    return [token for token in REQUIRED_SENTINELS if token not in vocab]


def _prepare_hf_tokenizer(tok):
    """Install the causal layout tokens and guarantee usable EOS/padding semantics."""
    vocab = tok.get_vocab()
    added = []

    # Some causal tokenizers have no declared EOS. Register the existing literal token when there
    # is one; otherwise add it as EOS (not merely as an unrelated additional special token).
    if tok.eos_token_id is None:
        if "<eos>" in vocab:
            tok.eos_token = "<eos>"
        else:
            tok.add_special_tokens({"eos_token": "<eos>"})
            added.append("<eos>")

    layout_tokens = [token for token in missing_sentinels(tok) if token != "<eos>"]  # EOS is above
    if layout_tokens:
        tok.add_special_tokens({"additional_special_tokens": layout_tokens})
        added.extend(layout_tokens)

    if tok.eos_token_id is None:
        raise ValueError("causal tokenizer has no eos_token_id after sentinel setup")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return added


def _hf_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mt)
    added = _prepare_hf_tokenizer(tok)
    if added:
        logger.warning(
            "🧩 %s lacks %s — added as new tokens. Their embeddings start RANDOM and the model has "
            "never seen this span-infilling format, so expect it to need more fine-tuning than a "
            "natively-infilling model like IgLM.", mt, added)
    adapter = TokenizerAdapter.from_hf(tok, position_type=pos, uses_special_tokens=wrap)
    adapter.mask_id = tok.convert_tokens_to_ids("<mask>")
    adapter.chain_sep_id = tok.convert_tokens_to_ids("<sep>")
    adapter.eos_id = tok.eos_token_id
    adapter.span_layout = validate_layout(CDRSEQGEN_CAUSAL, adapter)
    return adapter


def _load_hf_causal(name_or_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"🧬 Loading Hugging Face causal LM: {name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    _prepare_hf_tokenizer(tok)
    if model.get_input_embeddings().num_embeddings != len(tok):
        # Resize to the tokenizer the adapter will build, or the sentinel ids index out of bounds.
        model.resize_token_embeddings(len(tok))
    model.config.eos_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    if Path(name_or_path).is_dir():
        adapter = TokenizerAdapter.from_hf(tok, position_type="absolute", uses_special_tokens=True)
        adapter.mask_id = tok.convert_tokens_to_ids("<mask>")
        adapter.chain_sep_id = tok.convert_tokens_to_ids("<sep>")
        adapter.eos_id = tok.eos_token_id
        validate_checkpoint_layout(name_or_path, adapter)
    return model


register(CausalBackbone(
    id="hf_decoder",
    matches=lambda mt: "/" in mt or mt.startswith("."),
    position_type="absolute",
    uses_special_tokens=True,
    load_tokenizer=_hf_tokenizer,
    load_training_model=lambda mt, **kw: _load_hf_causal(mt),
    load_pretrained_model=lambda path: _load_hf_causal(path),
    fallback=True,
))


# The generic fallback must stay last, or a named backbone is silently shadowed.
_fallbacks = [i for i, b in enumerate(REGISTRY) if b.fallback]
assert _fallbacks == [len(REGISTRY) - 1], (
    f"the generic fallback must be the LAST registered causal backbone; "
    f"got fallbacks at {_fallbacks} of {[b.id for b in REGISTRY]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} causal backbones, fallback last: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "nferruz/ProtGPT2", "hugohrban/progen2-small", "./local_run"]:
        print(f"  {mt:<28} -> {resolve(mt).id}")
