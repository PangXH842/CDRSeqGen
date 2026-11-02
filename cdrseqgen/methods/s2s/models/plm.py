"""Protein encoder + the same residue decoder used by scratch S2S.

The PLM reads its native vocabulary; the decoder always emits CDRSeqGen's 27-token vocabulary.
An explicit projection joins the two widths. Only the encoder has pretrained weights. Checkpoints
embed its full HF config and state, so reloading never fetches the original model from the Hub.
"""
import logging

import torch
from torch import nn
from transformers import BertConfig, BertModel, EsmConfig, EsmModel, PreTrainedModel

from cdrseqgen.methods.common.transformer import (
    validate_architecture, validate_positive_integer, validate_transformer_config,
)
from .transformer import Seq2SeqAntibodyConfig, Seq2SeqAntibodyModel

logger = logging.getLogger(__name__)
ENCODERS = {"esm2": (EsmConfig, EsmModel), "protein_bert": (BertConfig, BertModel)}


class PLMSeq2SeqConfig(Seq2SeqAntibodyConfig):
    model_type = "plm_seq2seq_antibody"

    def __init__(self, encoder_config=None, encoder_family="esm2", encoder_name="",
                 encoder_init="pretrained", freeze_encoder=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.encoder_family = encoder_family
        self.encoder_name = encoder_name
        self.encoder_init = encoder_init
        self.freeze_encoder = freeze_encoder


def validate_encoder_options(encoder_init, freeze_encoder, num_encoder_layers):
    if encoder_init not in {"pretrained", "random"}:
        raise ValueError("encoder_init must be 'pretrained' or 'random'")
    if not isinstance(freeze_encoder, bool):
        raise ValueError("freeze_encoder must be a boolean")
    if num_encoder_layers is not None:
        raise ValueError("num_encoder_layers is fixed by the PLM checkpoint; leave it unset. "
                         "num_decoder_layers controls the new decoder.")


class PLMSeq2SeqModel(Seq2SeqAntibodyModel):
    config_class = PLMSeq2SeqConfig

    def __init__(self, config, encoder=None):
        validate_transformer_config(config, rotary=True, encoder_decoder=True)
        # Reuse the scratch decoder's layers, forward, aligned loss, and sampling interface.
        # Initialize ONLY new modules before attaching the encoder, so post_init cannot reset
        # pretrained parameters (or apply our initialization rules to HF-specific modules).
        PreTrainedModel.__init__(self, config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self._build_decoder(config)
        self.encoder_projection = nn.Linear(config.encoder_config["hidden_size"], config.hidden_size)
        self.post_init()
        if encoder is None:
            config_cls, model_cls = ENCODERS[config.encoder_family]
            encoder = model_cls(config_cls.from_dict(config.encoder_config), add_pooling_layer=False)
        self.encoder = encoder
        self.encoder.requires_grad_(not config.freeze_encoder)
        if config.freeze_encoder:
            self.encoder.eval()

    def save_pretrained(self, *args, **kwargs):
        # This composite owns its checkpoint format. Transformers 5 otherwise reverses nested
        # PLM LayerNorm names to legacy gamma/beta, which breaks strict reload and Trainer's
        # load_best_model_at_end. Preserve the actual state_dict keys on every save.
        kwargs["save_original_format"] = False
        return super().save_pretrained(*args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        # A frozen feature extractor must also disable dropout when Trainer calls model.train().
        if self.config.freeze_encoder:
            self.encoder.eval()
        return self

    def encode(self, input_ids, attention_mask=None):
        # ESM's token-dropout rescaling needs a mask even for an unpadded single example.
        if attention_mask is None:
            attention_mask = input_ids.ne(self.encoder.config.pad_token_id).long()
        if self.config.freeze_encoder:
            with torch.no_grad():
                hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                      return_dict=True).last_hidden_state
        else:
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                  return_dict=True).last_hidden_state
        return self.encoder_projection(hidden)


def build_model(model_type, *, encoder_family, vocab_size=27, hidden_size=128, num_layers=4,
                num_decoder_layers=None, num_encoder_layers=None, num_attention_heads=8,
                intermediate_size=None, max_len=512, encoder_init="pretrained",
                freeze_encoder=False, **_ignored):
    validate_encoder_options(encoder_init, freeze_encoder, num_encoder_layers)
    validate_architecture(
        hidden_size, num_layers, num_attention_heads, intermediate_size, max_len, rotary=True
    )
    validate_positive_integer("vocab_size", vocab_size)
    decoder_layers = num_layers if num_decoder_layers is None else num_decoder_layers
    validate_positive_integer("num_decoder_layers", decoder_layers)
    config_cls, model_cls = ENCODERS[encoder_family]
    raw_config, _ = config_cls.get_config_dict(model_type)
    if raw_config.get("model_type") not in (None, config_cls.model_type):
        raise ValueError(f"{model_type}: expected a {config_cls.model_type} encoder config, "
                         f"got {raw_config['model_type']!r}")
    encoder_config = config_cls.from_dict(raw_config)
    if encoder_config.is_decoder:
        raise ValueError("S2S requires a bidirectional PLM encoder, not a causal decoder")
    if encoder_family == "protein_bert" and max_len > encoder_config.max_position_embeddings:
        raise ValueError(f"max_len={max_len} exceeds the encoder position limit "
                         f"{encoder_config.max_position_embeddings}")
    if encoder_init == "pretrained":
        encoder, loading = model_cls.from_pretrained(
            model_type, config=encoder_config, add_pooling_layer=False, output_loading_info=True)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"{model_type}: pretrained encoder weights did not load completely: {loading}")
    else:
        encoder = model_cls(encoder_config, add_pooling_layer=False)
    config = PLMSeq2SeqConfig(
        encoder_config=encoder.config.to_dict(), encoder_family=encoder_family,
        encoder_name=str(model_type), encoder_init=encoder_init, freeze_encoder=freeze_encoder,
        vocab_size=vocab_size, hidden_size=hidden_size,
        num_hidden_layers=encoder.config.num_hidden_layers, num_decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=4 * hidden_size if intermediate_size is None else intermediate_size,
        max_position_embeddings=max(max_len, 512))
    model = PLMSeq2SeqModel(config, encoder=encoder)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("S2S %s encoder (%s, frozen=%s), residue decoder hidden=%s layers=%s: "
                "%s total / %s trainable parameters", model_type, encoder_init, freeze_encoder,
                hidden_size, decoder_layers, f"{total:,}", f"{trainable:,}")
    return model
