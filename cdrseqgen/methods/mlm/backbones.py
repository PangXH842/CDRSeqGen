"""Registry for masked-LM-compatible model backbones.

  * Backbone -- one model family's record: match predicate, encoding conventions, three loaders
  * register -- append a Backbone to REGISTRY (order matters, see below)
  * resolve  -- model_type string -> the first Backbone whose predicate matches

Current backbones:
  * scratch      -- the in-house transformer, models/transformer.py
  * esm2         -- Meta ESM-2; the only relative/rotary-position backbone
  * protein_bert -- IgBert / ProtBert / AntiBERTy; needs whitespace-separated residues
  * hf_encoder   -- generic AutoModelForMaskedLM fallback: any other hub id or local dir

Each backbone declares two conventions that the encoding depends on, so they are explicit rather
than guessed from the tokenizer:
  * position_type -- "absolute" | "relative". Where antigen context is injected (see
    AntibodyInfillingDataset). Absolute-position models (scratch, BERT-style) need it appended
    after the target; relative/rotary models (ESM-2) can take it prepended.
  * uses_special_tokens -- whether inputs are wrapped as [cls] ... [end].

Matching is by an ordered list of predicates; the first match wins, so register specific
backbones before the generic Hugging Face fallback.

--- To add a new PLM ---
Append one register(Backbone(...)) below (BEFORE the hf_encoder fallback if you need custom
conventions). Point load_tokenizer at its tokenizer, load_training_model/load_pretrained_model
at its HF model class, and set position_type correctly (absolute unless it uses rotary/relative
positions). Unless the generic hf_encoder fallback already fits it (any AutoModelForMaskedLM +
AutoTokenizer, absolute positions).

Note:
  * <model_cls>.from_pretrained(path) silently returns a *randomly initialized* model with
    the current transformers version — it builds the config but never loads the checkpoint weights
    (verified: lm_head norm came back at the std-0.02 init value, not the trained value), with no
    missing/unexpected-key warning. This broke both generate.py and any re-evaluation of scratch
    checkpoints. Load the state dict explicitly instead; keys match the freshly built model
    exactly, so strict=True is a hard correctness guard rather than a formality.
"""
import logging
from dataclasses import dataclass
from typing import Callable

from cdrseqgen.data.dataset_mlm import AntibodyTokenizer, TokenizerAdapter

logger = logging.getLogger(__name__)


@dataclass
class Backbone:
    id: str
    matches: Callable[[str], bool]
    position_type: str          # "absolute" | "relative"
    uses_special_tokens: bool
    load_tokenizer: Callable[[str, str, bool], TokenizerAdapter]  # (model_type, pos_type, wrap)
    load_training_model: Callable[..., object]                    # fresh model for training
    load_pretrained_model: Callable[[str], object]               # checkpoint for generation/eval
    fallback: bool = False

    def build_tokenizer(self, model_type) -> TokenizerAdapter:
        return self.load_tokenizer(model_type, self.position_type, self.uses_special_tokens)


REGISTRY = []


def register(backbone: Backbone) -> Backbone:
    REGISTRY.append(backbone)
    return backbone


def resolve(model_type: str) -> Backbone:
    """Return the first backbone whose predicate matches, else a helpful error."""
    for backbone in REGISTRY:
        if backbone.matches(model_type):
            return backbone
    raise ValueError(
        f"No backbone registered for model_type={model_type!r}. "
        f"Registered backbones: {[b.id for b in REGISTRY]}. Add one in backbones.py."
    )


# --- lazy loader helpers (imports kept local so the module is cheap to import) ----------------
def _load_custom_pretrained(path, model_cls, config_cls):
    import os
    from safetensors.torch import load_file
    config = config_cls.from_pretrained(path)
    model = model_cls(config)
    state_dict = load_file(os.path.join(path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

# --- custom from-scratch transformer -------------------------------------------
def _scratch():
    from cdrseqgen.methods.mlm.models import transformer
    return transformer


def _build_scratch(mt, vocab_size=25, hidden_size=128, num_layers=4,
                   num_attention_heads=8, intermediate_size=None, max_len=512, **_ignored):
    return _scratch().build_model(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_len=max_len,
    )


def _load_scratch_pretrained(path):
    t = _scratch()
    return _load_custom_pretrained(path, t.CustomAntibodyModel, t.CustomAntibodyConfig)


register(Backbone(
    id="scratch",
    matches=lambda mt: mt == "scratch",
    position_type="absolute",
    uses_special_tokens=False,
    load_tokenizer=lambda mt, pos, wrap: TokenizerAdapter.from_custom(AntibodyTokenizer(), position_type=pos),
    load_training_model=_build_scratch,
    load_pretrained_model=_load_scratch_pretrained,
))


# --- ESM-2 -------------------------------------------------
def _esm_tokenizer(mt, pos, wrap):
    from transformers import EsmTokenizer
    return TokenizerAdapter.from_hf(EsmTokenizer.from_pretrained(mt), position_type=pos, uses_special_tokens=wrap)


def _load_esm(name_or_path):
    from transformers import EsmForMaskedLM
    logger.info(f"🧬 Loading ESM-2 model: {name_or_path}")
    return EsmForMaskedLM.from_pretrained(name_or_path)


register(Backbone(
    id="esm2",
    matches=lambda mt: "esm2" in mt.lower(),   # .lower() to match protein_bert's convention
    position_type="relative",   # rotary embeddings -> target's absolute index doesn't matter
    uses_special_tokens=True,
    load_tokenizer=_esm_tokenizer,
    load_training_model=lambda mt, **kw: _load_esm(mt),
    load_pretrained_model=lambda path: _load_esm(path),
))


# --- antibody / protein BERT-style masked-LMs (IgBert, ProtBert, AntiBERTy, ...) --------------
def _bert_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer, BertTokenizer
    try:
        hf_tok = AutoTokenizer.from_pretrained(mt, use_fast=False)
    except Exception:
        hf_tok = BertTokenizer.from_pretrained(mt)
    return TokenizerAdapter.from_hf(hf_tok, position_type=pos, uses_special_tokens=wrap, space_residues=True)


def _load_bert_model(name_or_path):
    from transformers import AutoModelForMaskedLM, BertForMaskedLM
    logger.info(f"🧬 Loading protein-BERT masked-LM: {name_or_path}")
    try:
        return AutoModelForMaskedLM.from_pretrained(name_or_path)
    except (ValueError, KeyError):
        return BertForMaskedLM.from_pretrained(name_or_path)


register(Backbone(
    id="protein_bert",
    matches=lambda mt: "bert" in mt.lower(),   # IgBert, prot_bert, AntiBERTy, ...
    position_type="absolute",
    uses_special_tokens=True,
    load_tokenizer=_bert_tokenizer,
    load_training_model=lambda mt, **kw: _load_bert_model(mt),
    load_pretrained_model=lambda path: _load_bert_model(path),
))


# --- generic Hugging Face encoder fallback (AbLang2, ESM-C, ...) -------------------------------
def _hf_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer
    return TokenizerAdapter.from_hf(AutoTokenizer.from_pretrained(mt), position_type=pos, uses_special_tokens=wrap)


def _load_automodel(name_or_path):
    from transformers import AutoModelForMaskedLM
    logger.info(f"🧬 Loading Hugging Face masked-LM: {name_or_path}")
    return AutoModelForMaskedLM.from_pretrained(name_or_path)


register(Backbone(
    id="hf_encoder",
    matches=lambda mt: "/" in mt or mt.startswith("."),
    position_type="absolute",
    uses_special_tokens=True,
    load_tokenizer=_hf_tokenizer,
    load_training_model=lambda mt, **kw: _load_automodel(mt),
    load_pretrained_model=lambda path: _load_automodel(path),
    fallback=True,
))


# Assert fallback model (or else quietly resolves to hf_encoder)
_fallbacks = [i for i, b in enumerate(REGISTRY) if b.fallback]
assert _fallbacks == [len(REGISTRY) - 1], (
    f"the generic fallback must be the LAST registered backbone; "
    f"got fallbacks at {_fallbacks} of {[b.id for b in REGISTRY]}"
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} backbones registered, fallback last: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "facebook/esm2_t12_35M_UR50D", "Exscientia/IgBert",
               "Rostlab/prot_bert", "ablang/ablang2"]:
        print(f"  {mt:32s} -> {resolve(mt).id}")
    _scratch()  # raises here if the custom/ import path is wrong
    print("\n✅ dispatch + lazy imports OK")
