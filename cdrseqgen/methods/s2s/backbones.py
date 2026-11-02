"""Register S2S backbones: scratch, or a PLM encoder with a new residue decoder.

Shared API:
  resolve(model_type) -> Seq2SeqBackbone -> build_tokenizer(), load_training_model(),
  load_pretrained_model(). Seq2SeqBackbone names the shared common.backbones.Backbone
  descriptor; this module owns encoder/decoder assembly and tokenizer adaptation.

Functions:
  * register -- append a Seq2SeqBackbone to REGISTRY and return it
  * resolve  -- return the first matching Seq2SeqBackbone; raise ValueError if none match

Current backbones (first match wins):
  * scratch      -- custom encoder-decoder (models/transformer.py), rotary positions
  * esm2         -- name contains "esm2"; ESM-2 encoder, relative layout
  * protein_bert -- name contains "bert"; protein-BERT encoder, absolute layout
  Both PLM encoders feed the same new residue decoder (models/plm.py).

Notes:
  * ESM2 and protein BERTs reuse the shared protein tokenizer conventions. Their encoder inputs
    keep native gap/wrapper tokens; targets use the same residue vocabulary and EOS as scratch S2S.
  * There is no generic full encoder-decoder fallback. ProtT5 used BART-like residue masking, not
    original T5 span denoising; a future full ProtT5 adapter must validate its actual input/output
    contract instead of assuming native missing-span generation from the architecture name.
  * This registry permits no fallback. If one is added, the shared registry checks
    that it is last.
  * Running this module directly prints example resolutions and checks the scratch-model import;
    it does not load checkpoint weights or run training.
"""
import logging
from functools import partial

from cdrseqgen.datasets.dataset_s2s import Seq2SeqTokenizer, s2s_adapter
from cdrseqgen.methods.common.backbones import (
    Backbone as Seq2SeqBackbone, BackboneRegistry,
    bert_tokenizer, esm_tokenizer, load_custom_pretrained as _load_custom_pretrained)
from cdrseqgen.methods.common.transformer import validate_transformer_args

logger = logging.getLogger(__name__)


# --- (1) Method-local registry; shared descriptor and mechanics -----------------------------------
_registry = BackboneRegistry(
    "s2s",
    unsupported_hint="Supported PLM encoders are ESM2 and protein BERTs; "
                     "full pretrained encoder-decoders require their own adapter.",
)
REGISTRY = _registry.entries
register = _registry.register
resolve = _registry.resolve


# --- (2a) local model definitions: custom from-scratch encoder-decoder -----------------------------
def _validate_scratch_args(model_args, data_args):
    validate_transformer_args(model_args, data_args, rotary=True)
    if model_args.freeze_encoder or model_args.encoder_init != "pretrained":
        raise ValueError("freeze_encoder and encoder_init controls require a PLM backbone")


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
    validate_model_args=_validate_scratch_args,
))


# --- (2b) external PLM encoders: ESM-2 and protein BERTs, each with a new residue decoder ---------
def _validate_plm_args(model_args, data_args):
    from cdrseqgen.methods.s2s.models.plm import validate_encoder_options
    validate_transformer_args(model_args, data_args, rotary=True)
    validate_encoder_options(
        model_args.encoder_init, model_args.freeze_encoder, model_args.num_encoder_layers
    )


def _plm_tokenizer(mt, pos, wrap, *, family):
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
        validate_model_args=_validate_plm_args,
    ))


_registry.validate()


# --- MAIN ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"{len(REGISTRY)} seq2seq backbones registered: {[b.id for b in REGISTRY]}\n")
    for mt in ["scratch", "facebook/esm2_t12_35M_UR50D", "Rostlab/prot_bert",
               "Rostlab/prot_t5_xl_uniref50"]:
        try:
            print(f"  {mt:<32} -> {resolve(mt).id}")
        except ValueError as exc:
            print(f"  {mt:<32} -> unregistered ({str(exc).splitlines()[0][:60]}…)")
    _scratch()  # raises here if the models/ import path is wrong
    print("\n✅ dispatch + lazy imports OK")
