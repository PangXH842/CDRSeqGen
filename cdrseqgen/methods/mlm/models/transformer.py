"""Custom bidirectional Transformer for masked-loop reconstruction.

Structure:
  CustomAntibodyConfig -> CustomAntibodyModel -> build_model()
  token + learned absolute positions -> encoder -> residue logits [B, S, V]

Contract:
  * Same-position labels; -100 excludes unsupervised tokens.
  * Optional hidden_states contains the final encoder representation.
  * Training uses MLMLossTrainer for the configured external objective.
  * return_dict is accepted for HF compatibility; outputs are always dictionaries.

Builders accept intermediate_size=None for 4 x hidden_size and allocate at least
512 learned position slots. Backbone registrations own selection and strict
checkpoint restoration; method CLI entrypoints configure logging.
"""

import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from cdrseqgen.methods.common.transformer import (
    init_transformer_weights,
    validate_architecture,
    validate_transformer_config,
)

logger = logging.getLogger(__name__)


class CustomAntibodyConfig(PretrainedConfig):
    """Persist resolved architecture dimensions under the existing checkpoint model_type."""

    model_type = "custom_antibody_transformer"

    def __init__(
        self,
        vocab_size=25,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings


class CustomAntibodyModel(PreTrainedModel):
    """Bidirectional encoder with learned positions and optional reconstruction loss."""

    config_class = CustomAntibodyConfig

    def __init__(self, config):
        validate_transformer_config(config, rotary=False)
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Tells Hugging Face to finalize the model setup (including the missing 'all_tied_weights_keys')
        self.post_init()

    # Hugging Face requires custom models to define how they should be initialized
    def _init_weights(self, module):
        init_transformer_weights(module)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, return_dict=True, **_kwargs):
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # attention_mask is 1=real token / 0=padding; TransformerEncoder's
        # src_key_padding_mask wants the opposite convention (True = ignore this position).
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # ignore_index=-100 is default
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        output = {"loss": loss, "logits": logits}
        if output_hidden_states:
            output["hidden_states"] = (x,)
        return output


def build_model(vocab_size=25, hidden_size=128, num_layers=4, num_attention_heads=8,
                intermediate_size=None, max_len=512):
    """Build fresh weights after validating dimensions; None FFN width means 4 x hidden_size."""
    validate_architecture(
        hidden_size, num_layers, num_attention_heads, intermediate_size, max_len,
        rotary=False,
    )
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    logger.info(f"🚀 Initializing Custom Transformer from scratch "
                f"(hidden={hidden_size}, layers={num_layers}, heads={num_attention_heads}, "
                f"ffn={intermediate_size})")
    config = CustomAntibodyConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max(max_len, 512),
    )
    return CustomAntibodyModel(config)
