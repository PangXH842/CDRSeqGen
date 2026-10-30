"""Register MLM tokenizers and model loaders using common.backbones.

Shared API:
  resolve(model_type) -> Backbone -> build_tokenizer(), load_training_model(),
  load_pretrained_model(). The descriptor and registry mechanics live in common;
  this module owns the masked-LM loaders and ordered registration blocks.

Current backbones (first match wins):
  * scratch      -- custom transformer and tokenizer
  * esm2         -- name contains "esm2"; ESM tokenizer, relative layout
  * protein_bert -- name contains "bert"; whitespace-separated residue encoding
  * hf_encoder   -- remaining names containing "/" or starting with ".";
                    AutoTokenizer + AutoModelForMaskedLM, absolute layout

Extending:
  * Add a register(Backbone(...)) before the generic fallback when custom
    tokenization, layout conventions or model loading are required.
  * Reuse the fallback only when its masked-LM loader, residue encoding and
    layout conventions fit the model. Matching a name does not verify these.
  * The fallback must remain last; the shared registry checks this at import.

Notes:
  * Scratch checkpoints use common.backbones.load_custom_pretrained():
    rebuild from the saved config and load model.safetensors with strict=True.
  * Running this module directly prints example resolutions and checks the
    scratch-model import; it does not load checkpoint weights or run training.
"""
import logging

from cdrseqgen.datasets.dataset_common import TokenizerAdapter
from cdrseqgen.datasets.dataset_mlm import AntibodyTokenizer

from cdrseqgen.methods.common.backbones import (
    Backbone,
    BackboneRegistry,
    load_custom_pretrained as _load_custom_pretrained,
    esm_tokenizer as _esm_tokenizer, bert_tokenizer as _bert_tokenizer)

logger = logging.getLogger(__name__)

# --- (1) Method-local registry; shared descriptor and mechanics -----------------------------------
_registry = BackboneRegistry("mlm", require_fallback=True)
REGISTRY = _registry.entries
register = _registry.register
resolve = _registry.resolve


# --- (2a) local model definitions ------
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


# --- (2b) external PLMs ------
# --- ESM-2 -------------------------------------------------

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


_registry.validate()

# --- MAIN ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} backbones registered, fallback last: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "facebook/esm2_t12_35M_UR50D", "Exscientia/IgBert",
               "Rostlab/prot_bert", "ablang/ablang2"]:
        print(f"  {mt:32s} -> {resolve(mt).id}")
    _scratch()  # raises here if the custom/ import path is wrong
    print("\n✅ dispatch + lazy imports OK")
