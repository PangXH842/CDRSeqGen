"""S2S backbones: scratch, or a protein PLM encoder with a new residue decoder.

ESM2 and protein BERTs reuse the shared protein tokenizer conventions. Their encoder inputs keep
native gap/wrapper tokens; targets use the same residue vocabulary and EOS as scratch S2S.
There is no generic full encoder-decoder fallback. ProtT5 used BART-like residue masking, not
original T5 span denoising; a future full ProtT5 adapter must validate its actual input/output
contract instead of assuming native missing-span generation from the architecture name.
"""
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable

from cdrseqgen.data.dataset_common import TokenizerAdapter
from cdrseqgen.data.dataset_s2s import Seq2SeqTokenizer, s2s_adapter

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqBackbone:
    id: str
    matches: Callable[[str], bool]
    position_type: str          # "relative" for rotary/relative models, "absolute" otherwise
    uses_special_tokens: bool
    load_tokenizer: Callable[[str, str, bool], TokenizerAdapter]
    load_training_model: Callable[..., object]
    load_pretrained_model: Callable[[str], object]
    fallback: bool = False

    def build_tokenizer(self, model_type, checkpoint_path=None) -> TokenizerAdapter:
        from cdrseqgen.methods.common.backbones import tokenizer_source
        source = tokenizer_source(model_type, checkpoint_path)
        return self.load_tokenizer(source, self.position_type, self.uses_special_tokens)


REGISTRY = []


def register(backbone: Seq2SeqBackbone) -> Seq2SeqBackbone:
    REGISTRY.append(backbone)
    return backbone


def resolve(model_type: str) -> Seq2SeqBackbone:
    """Return the first seq2seq backbone whose predicate matches, else a helpful error."""
    for backbone in REGISTRY:
        if backbone.matches(model_type):
            return backbone
    raise ValueError(
        f"No seq2seq backbone registered for model_type={model_type!r}. "
        f"Registered backbones: {[b.id for b in REGISTRY]}. Supported PLM encoders are ESM2 and "
        f"protein BERTs; full pretrained encoder-decoders require their own adapter.")


from cdrseqgen.methods.common.backbones import load_custom_pretrained as _load_custom_pretrained


# --- custom from-scratch encoder-decoder ---------------------------------------------------------
def _scratch():
    from cdrseqgen.methods.s2s.models import transformer
    return transformer


def _build_scratch(mt, vocab_size=None, hidden_size=128, num_layers=4, num_attention_heads=8,
                   intermediate_size=None, max_len=512, num_encoder_layers=None,
                   num_decoder_layers=None, **_ignored):
    # Default to the generative vocab (27) rather than the masked one (25): the two extra rows are
    # <bos>/<eos>, and sizing to 25 would make <eos> unrepresentable.
    if vocab_size is None:
        vocab_size = len(Seq2SeqTokenizer().vocab)
    return _scratch().build_model(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
        num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
        max_len=max_len, num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers)


def _load_scratch_pretrained(path):
    t = _scratch()
    return _load_custom_pretrained(path, t.Seq2SeqAntibodyModel, t.Seq2SeqAntibodyConfig)


register(Seq2SeqBackbone(
    id="scratch",
    matches=lambda mt: mt == "scratch",
    position_type="relative",       # rotary — see models/transformer.py and Findings §8
    uses_special_tokens=False,
    load_tokenizer=lambda mt, pos, wrap: s2s_adapter(position_type=pos),
    load_training_model=_build_scratch,
    load_pretrained_model=_load_scratch_pretrained,
))


def _plm_tokenizer(mt, pos, wrap, *, family):
    from cdrseqgen.methods.common.backbones import esm_tokenizer, bert_tokenizer
    loader = esm_tokenizer if family == "esm2" else bert_tokenizer
    encoder = loader(mt, pos, wrap)
    for name in ("pad_id", "mask_id", "cls_id", "end_id", "chain_sep_id"):
        if getattr(encoder, name) is None:
            raise ValueError(f"{mt}: encoder tokenizer requires {name}")
    residues = "ACDEFGHIKLMNPQRSTVWY"
    ids = encoder.encode(residues)
    if (len(ids) != len(residues) or len(set(ids)) != len(residues)
            or set(ids) & encoder.special_ids or encoder.decode(ids) != residues):
        raise ValueError(f"{mt}: encoder tokenizer must preserve canonical residues one token each")
    adapter = s2s_adapter(position_type=pos)
    adapter.encoder_tokenizer = encoder
    adapter.save_pretrained = encoder.save_pretrained
    return adapter


def _build_plm(mt, *, encoder_family, **kwargs):
    from cdrseqgen.methods.s2s.models.plm import build_model
    return build_model(mt, encoder_family=encoder_family, **kwargs)


def _load_plm_pretrained(path):
    from cdrseqgen.methods.s2s.models.plm import PLMSeq2SeqConfig, PLMSeq2SeqModel
    return _load_custom_pretrained(path, PLMSeq2SeqModel, PLMSeq2SeqConfig)


for family, match, position in (
    ("esm2", lambda mt: "esm2" in mt.lower(), "relative"),
    ("protein_bert", lambda mt: "bert" in mt.lower(), "absolute"),
):
    register(Seq2SeqBackbone(
        id=family, matches=match, position_type=position, uses_special_tokens=True,
        load_tokenizer=partial(_plm_tokenizer, family=family),
        load_training_model=partial(_build_plm, encoder_family=family),
        load_pretrained_model=_load_plm_pretrained,
    ))


# A generic fallback, once one exists, must stay last or a named backbone is silently shadowed.
_fallbacks = [i for i, b in enumerate(REGISTRY) if b.fallback]
assert _fallbacks in ([], [len(REGISTRY) - 1]), (
    f"the generic fallback must be the LAST registered seq2seq backbone; "
    f"got fallbacks at {_fallbacks} of {[b.id for b in REGISTRY]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} seq2seq backbones: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "Rostlab/prot_t5_xl_uniref50"]:
        try:
            print(f"  {mt:<32} -> {resolve(mt).id}")
        except ValueError as exc:
            print(f"  {mt:<32} -> unregistered ({str(exc).splitlines()[0][:60]}…)")
