"""Register AR backbones and enforce their span-infilling checkpoint contract.

Shared API:
  resolve(model_type) -> CausalBackbone -> build_tokenizer(), load_training_model(),
  load_pretrained_model(). common.backbones supplies the descriptor and registry.
  transformer.py and iglm.py each supply load_tokenizer, load_training_model,
  load_pretrained_model and validate_training_args; this registry wires them lazily.

AR extension:
  CausalBackbone.build_tokenizer() loads through the shared implementation, then
  validates the tokenizer/checkpoint span layout and attaches layout persistence.

Functions:
  * register -- append a CausalBackbone to REGISTRY and return it
  * resolve  -- return the first matching CausalBackbone; raise ValueError if none match
  * validate_checkpoint_layout, persist_layout_with_tokenizer, record_training_layout
             -- the span-layout contract a checkpoint carries
  * missing_sentinels -- which layout sentinels a pretrained tokenizer lacks

Current backbones (first match wins):
  * scratch    -- custom causal decoder (models/transformer.py), rotary positions
  * iglm       -- native IgLM / IgLM-S infilling (models/iglm.py), original 33-token vocabulary,
                  its own layout and sampler
  * hf_decoder -- remaining names containing "/" or starting with "."; AutoModelForCausalLM,
                  absolute layout

Extending:
  * Add a register(CausalBackbone(...)) before the generic fallback; entrypoints never branch on
    model type.
  * The fallback must remain last; the shared registry checks this at import.
  * A backbone with its own span format attaches span_layout (and, if it restricts decoding,
    causal_sampler and causal_sampling_metadata) to its tokenizer adapter. A backbone whose
    original release carries no layout metadata supplies native_checkpoint, the predicate that
    recognizes that release; every other checkpoint must keep its recorded layout.

Notes:
  * Pretrained causal tokenizers receive any missing gap/separator/terminator
    tokens. Their new embeddings start random; the model must learn this format
    during fine-tuning. The added tokens and vocabulary resizing are AR-owned.
  * A native-format backbone (iglm) adds no tokens and never resizes embeddings.
  * Running this module directly prints example resolutions and checks the scratch-model import;
    it does not load checkpoint weights or run training.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cdrseqgen.datasets.dataset_ar import (CDRSEQGEN_CAUSAL,
                                         layout_for, validate_layout, span_layout_metadata)
from cdrseqgen.artifacts import atomic_write_json, find_run_config
from cdrseqgen.datasets.dataset_common import TokenizerAdapter
from cdrseqgen.methods.common.backbones import (
    Backbone, BackboneRegistry)

logger = logging.getLogger(__name__)

LAYOUT_FILENAME = "span_layout.json"


# --- span-layout contract of a checkpoint (AR only) -----------------------------------------------
def validate_checkpoint_layout(path, tokenizer, *, native_checkpoint=False):
    """All recorded contracts must agree with the live tokenizer/layout; absent legacy is explicit.

    native_checkpoint marks an original upstream release (no CDRSeqGen metadata exists to record);
    it is set only by a backbone's own recognition predicate, never inferred from a missing file.
    """
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
            if ('causal_sampling' in payload
                    and payload['causal_sampling'] != getattr(tokenizer, 'causal_sampling_metadata', None)):
                raise ValueError(f"Causal sampling policy mismatch in {source}; "
                                 "checkpoint decoding cannot be silently reinterpreted")
    for source, actual in recorded:
        if actual != expected or any(type(actual[key]) is not type(value)
                                     for key, value in expected.items()):
            raise ValueError(f"Span layout mismatch in {source}: recorded {actual!r}, "
                             f"requested {expected!r}. Layout reinterpretation is not supported.")
    if not recorded:
        if native_checkpoint:
            logger.info("Importing recognized native checkpoint %s with layout %s version %s",
                        path, expected['id'], expected['version'])
            return expected
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
    sampling = getattr(tokenizer, 'causal_sampling_metadata', None)
    model.config.span_layout = metadata
    if sampling is not None:
        model.config.causal_sampling = dict(sampling)
    source = find_run_config(output_dir)
    if source is None:
        raise ValueError("Training layout requires an initialized run_config.json")
    with Path(source).open() as handle:
        run_config = json.load(handle)
    run_config['span_layout'] = metadata
    if sampling is not None:
        run_config['causal_sampling'] = dict(sampling)
    atomic_write_json(source, run_config)


# The sentinels the causal layout needs beyond a plain LM vocabulary.
REQUIRED_SENTINELS = ("<mask>", "<sep>", "<eos>")


# --- (1) Method-local descriptor and registry; shared mechanics ------------------------------------
@dataclass
class CausalBackbone(Backbone):
    """Extend tokenizer construction with AR span-layout validation and saving."""

    native_checkpoint: Callable[[str], bool] | None = None

    def build_tokenizer(self, model_type, checkpoint_path=None) -> TokenizerAdapter:
        tokenizer = super().build_tokenizer(model_type, checkpoint_path)
        validate_layout(layout_for(tokenizer), tokenizer)
        source = checkpoint_path or getattr(tokenizer, "checkpoint_source", None)
        if source:
            native = self.native_checkpoint is not None and self.native_checkpoint(source)
            validate_checkpoint_layout(source, tokenizer, native_checkpoint=native)
        return persist_layout_with_tokenizer(tokenizer)


_registry = BackboneRegistry("ar", require_fallback=True)
REGISTRY = _registry.entries
register = _registry.register
resolve = _registry.resolve


# --- (2a) local model definitions: custom from-scratch causal decoder -----------------------------
def _scratch():
    from cdrseqgen.methods.ar.models import transformer
    return transformer


register(CausalBackbone(
    id="scratch",
    matches=lambda mt: mt == "scratch",
    position_type="relative",       # rotary — see models/transformer.py and Findings §8
    uses_special_tokens=False,
    load_tokenizer=lambda mt, pos, wrap: _scratch().load_tokenizer(mt, pos, wrap),
    load_training_model=lambda mt, **kw: _scratch().load_training_model(mt, **kw),
    load_pretrained_model=lambda path: _scratch().load_pretrained_model(path),
    validate_model_args=lambda model, data: _scratch().validate_training_args(model, data),
))


# --- (2b) external PLMs: native IgLM infilling, then the generic Hugging Face causal LM fallback ----
def _iglm():
    from cdrseqgen.methods.ar.models import iglm
    return iglm


register(CausalBackbone(
    id="iglm",
    matches=lambda mt: _iglm().matches(mt),
    position_type="absolute",
    uses_special_tokens=False,      # chain/species and markers are supplied by the SpanLayout
    load_tokenizer=lambda mt, pos, wrap: _iglm().load_tokenizer(mt, pos, wrap),
    load_training_model=lambda mt, **kw: _iglm().load_training_model(mt, **kw),
    load_pretrained_model=lambda path: _iglm().load_pretrained_model(path),
    validate_model_args=lambda model, data: _iglm().validate_training_args(model, data),
    native_checkpoint=lambda path: _iglm().is_native_checkpoint(path),
))


# Generic Hugging Face causal LMs, last: any "/" or "." name that no entry above claims.
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


_registry.validate()


# --- MAIN ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} causal backbones registered, fallback last: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "IgLM", "IgLM-S", "nferruz/ProtGPT2", "hugohrban/progen2-small",
               "./local_run"]:
        print(f"  {mt:<28} -> {resolve(mt).id}")
    _scratch()  # raises here if the models/ import path is wrong
    print("\n✅ dispatch + lazy imports OK")
