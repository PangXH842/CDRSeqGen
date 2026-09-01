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
import logging
from dataclasses import dataclass
from typing import Callable

from cdrseqgen.data.dataset_ar import CausalTokenizer, causal_adapter
from cdrseqgen.data.dataset_common import TokenizerAdapter

logger = logging.getLogger(__name__)

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

    def build_tokenizer(self, model_type) -> TokenizerAdapter:
        return self.load_tokenizer(model_type, self.position_type, self.uses_special_tokens)


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


def _load_custom_pretrained(path, model_cls, config_cls):
    import os
    from safetensors.torch import load_file
    config = config_cls.from_pretrained(path)
    model = model_cls(config)
    model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=True)
    model.eval()
    return model


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


def _hf_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mt)
    added = missing_sentinels(tok)
    if added:
        # The model must be resized to match; _load_hf_causal below does that from the same list.
        tok.add_special_tokens({"additional_special_tokens": added})
        logger.warning(
            "🧩 %s lacks %s — added as new tokens. Their embeddings start RANDOM and the model has "
            "never seen this span-infilling format, so expect it to need more fine-tuning than a "
            "natively-infilling model like IgLM.", mt, added)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token          # HF causal LMs commonly ship without a pad token
    adapter = TokenizerAdapter.from_hf(tok, position_type=pos, uses_special_tokens=wrap)
    adapter.mask_id = tok.convert_tokens_to_ids("<mask>")
    adapter.chain_sep_id = tok.convert_tokens_to_ids("<sep>")
    adapter.eos_id = tok.eos_token_id
    return adapter


def _load_hf_causal(name_or_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"🧬 Loading Hugging Face causal LM: {name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(name_or_path)
    tok = AutoTokenizer.from_pretrained(name_or_path)
    added = missing_sentinels(tok)
    if added:
        # Resize to the tokenizer the adapter will build, or the sentinel ids index out of bounds.
        model.resize_token_embeddings(len(tok) + len(added))
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
