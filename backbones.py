"""Backbone registry — the single place to teach CDRSeqGen a new model.

Adding a protein language model (AbLang, ProtBert, AntiBERTy, ESM-C, ...) is one `register(...)`
call here. Nothing else in the codebase branches on model family: `train.py`, `predict.py`, and
`dataset.py` all go through `resolve(model_type)` and the `TokenizerAdapter` it hands back.

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
positions). If the generic hf_encoder fallback already fits it (any AutoModelForMaskedLM +
AutoTokenizer, absolute positions), you don't even need a new entry — it will just work.
"""
from dataclasses import dataclass
from typing import Callable

from dataset import AntibodyTokenizer, TokenizerAdapter
from model import build_scratch_model


@dataclass
class Backbone:
    id: str
    matches: Callable[[str], bool]
    position_type: str          # "absolute" | "relative"
    uses_special_tokens: bool
    load_tokenizer: Callable[[str, str, bool], TokenizerAdapter]  # (model_type, pos_type, wrap)
    load_training_model: Callable[..., object]                    # fresh model for training
    load_pretrained_model: Callable[[str], object]               # saved checkpoint for inference

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


# --- scratch: the in-house from-scratch transformer -------------------------------------------
register(Backbone(
    id="scratch",
    matches=lambda mt: mt == "scratch",
    position_type="absolute",
    uses_special_tokens=False,
    load_tokenizer=lambda mt, pos, wrap: TokenizerAdapter.from_custom(AntibodyTokenizer(), position_type=pos),
    load_training_model=lambda mt, vocab_size=25, hidden_size=128, num_layers=4, max_len=512:
        build_scratch_model(vocab_size, hidden_size, num_layers, max_len),
    load_pretrained_model=lambda path: _load_custom_pretrained(path),
))


# --- ESM-2: Meta's rotary-position protein LM -------------------------------------------------
def _esm_tokenizer(mt, pos, wrap):
    from transformers import EsmTokenizer
    return TokenizerAdapter.from_hf(EsmTokenizer.from_pretrained(mt), position_type=pos, uses_special_tokens=wrap)


register(Backbone(
    id="esm2",
    matches=lambda mt: "esm2" in mt,
    position_type="relative",   # rotary embeddings -> target's absolute index doesn't matter
    uses_special_tokens=True,
    load_tokenizer=_esm_tokenizer,
    load_training_model=lambda mt, **kw: _load_esm(mt),
    load_pretrained_model=lambda path: _load_esm(path),
))


# --- antibody / protein BERT-style masked-LMs (IgBert, ProtBert, AntiBERTy, ...) --------------
# BERT-family models trained on protein/antibody sequences. Two differences from the generic
# fallback below force a dedicated entry: (1) their tokenizers expect whitespace-separated
# residues ("M E V Q"), handled via space_residues=True; (2) they have ABSOLUTE positions, so
# antigen context is appended after the target (same as the scratch model). IgBert (Exscientia,
# trained on OAS *paired* repertoires) is the antibody-specific option and the reason this exists;
# ProtBert (Rostlab) is the general-protein baseline. Registered ABOVE hf_encoder because these
# hub ids also contain "/" and would otherwise be grabbed by the fallback WITHOUT residue spacing.
def _bert_tokenizer(mt, pos, wrap):
    # AutoTokenizer insists on building a *fast* tokenizer for these protein-BERT repos (ProtBert,
    # IgBert) and fails in current transformers even with sentencepiece installed; the concrete
    # slow BertTokenizer loads the WordPiece vocab.txt fine. Try Auto first (covers models that do
    # ship a fast tokenizer), then fall back to the slow BERT class.
    from transformers import AutoTokenizer, BertTokenizer
    try:
        hf_tok = AutoTokenizer.from_pretrained(mt, use_fast=False)
    except Exception:
        hf_tok = BertTokenizer.from_pretrained(mt)
    return TokenizerAdapter.from_hf(hf_tok, position_type=pos, uses_special_tokens=wrap, space_residues=True)


def _load_bert_model(name_or_path):
    # Some protein-BERT repos (e.g. Rostlab/prot_bert) ship a config.json with NO model_type key,
    # so AutoModelForMaskedLM/AutoConfig can't resolve them ("Unrecognized model") even though the
    # architecture is plainly BertForMaskedLM. Try Auto first (works for IgBert etc.), then fall
    # back to the concrete BertForMaskedLM class.
    from transformers import AutoModelForMaskedLM, BertForMaskedLM
    print(f"🧬 Loading protein-BERT masked-LM: {name_or_path}")
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
# Any hub path or local dir that isn't scratch/ESM-2/BERT. Assumes an AutoModelForMaskedLM with an
# AutoTokenizer and ABSOLUTE positions (the common case for BERT/RoBERTa-style protein LMs). A PLM
# that needs different handling (e.g. relative positions) should get its own entry ABOVE this one.
def _hf_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer
    return TokenizerAdapter.from_hf(AutoTokenizer.from_pretrained(mt), position_type=pos, uses_special_tokens=wrap)


register(Backbone(
    id="hf_encoder",
    matches=lambda mt: "/" in mt or mt.startswith("."),  # looks like a hub path or local dir
    position_type="absolute",
    uses_special_tokens=True,
    load_tokenizer=_hf_tokenizer,
    load_training_model=lambda mt, **kw: _load_automodel(mt),
    load_pretrained_model=lambda path: _load_automodel(path),
))


# --- lazy loader helpers (imports kept local so the module is cheap to import) ----------------
def _load_custom_pretrained(path):
    # NOTE: CustomAntibodyModel.from_pretrained(path) silently returns a *randomly initialized*
    # model with the current transformers version — it builds the config but never loads the
    # checkpoint weights (verified: lm_head norm came back at the std-0.02 init value, not the
    # trained value), with no missing/unexpected-key warning. This broke both predict.py and any
    # re-evaluation of scratch checkpoints. Load the state dict explicitly instead; keys match the
    # freshly built model exactly, so strict loading is a hard correctness guard.
    import os
    from safetensors.torch import load_file
    from model import CustomAntibodyModel, CustomAntibodyConfig
    config = CustomAntibodyConfig.from_pretrained(path)
    model = CustomAntibodyModel(config)
    state_dict = load_file(os.path.join(path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_esm(name_or_path):
    from transformers import EsmForMaskedLM
    print(f"🧬 Loading ESM-2 model: {name_or_path}")
    return EsmForMaskedLM.from_pretrained(name_or_path)


def _load_automodel(name_or_path):
    from transformers import AutoModelForMaskedLM
    print(f"🧬 Loading Hugging Face masked-LM: {name_or_path}")
    return AutoModelForMaskedLM.from_pretrained(name_or_path)
