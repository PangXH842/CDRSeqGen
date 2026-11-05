"""Define and load the custom causal Transformer for AR loop infilling.

Components:
  * CausalAntibodyConfig / CausalAntibodyModel -- architecture settings and model
  * CausalSelfAttention / CausalBlock         -- rotary attention and pre-norm block
  * build_model                              -- validate dimensions and create fresh weights
  * Registry hooks: load_tokenizer, load_training_model, load_pretrained_model,
                    validate_training_args (the same names as models/iglm.py)

Flow:
  backbones -> load_training_model -> build_model -> Config -> Model
  input_ids -> embedding -> causal rotary blocks -> norm -> logits [B, S, V]

Contract:
  * Labels share input positions; logits[t] predicts labels[t + 1]. -100 is ignored.
  * Loss is a token mean, or a sum divided by a supplied num_items_in_batch.
  * Outputs are dictionaries: loss, logits, and optional final hidden_states.
    return_dict is accepted without changing that format.
"""
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from cdrseqgen.methods.common.transformer import (
    RotarySelfAttention,
    feed_forward,
    init_transformer_weights,
    validate_architecture,
    validate_transformer_args,
    validate_transformer_config,
)

logger = logging.getLogger(__name__)


# --- Architecture ------------------------------------------------------------------------------
class CausalAntibodyConfig(PretrainedConfig):
    """Persist resolved architecture dimensions under the existing checkpoint model_type."""

    model_type = "causal_antibody_transformer"

    def __init__(self, vocab_size=27, hidden_size=128, num_hidden_layers=4,
                 num_attention_heads=8, intermediate_size=512, max_position_embeddings=512,
                 rope_theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta


class CausalSelfAttention(RotarySelfAttention):
    """Use shared rotary self-attention with the causal mask always enabled."""

    def __init__(self, config):
        super().__init__(config, causal=True)


class CausalBlock(nn.Module):
    """Apply pre-norm causal attention and FFN, each with a residual connection."""

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = feed_forward(config)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.mlp(self.norm2(x))


class CausalAntibodyModel(PreTrainedModel):
    """Causal rotary stack with next-token loss and optional final hidden states."""

    config_class = CausalAntibodyConfig

    def __init__(self, config):
        validate_transformer_config(config, rotary=True)
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        init_transformer_weights(module)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, return_dict=True, num_items_in_batch=None, **_kwargs):
        """Return loss/logits and optionally the final hidden representation.

        Example: inputs P P A C EOS, labels -100 -100 A C EOS
          logits at positions 1, 2, 3 predict A, C, EOS respectively.

        Trainer supplies num_items_in_batch for the whole accumulation window;
        divide the loss sum by that count. Direct calls use token-mean CE.
        """
        x = self.embedding(input_ids)
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1),
                reduction="mean" if num_items_in_batch is None else "sum")
            if num_items_in_batch is not None:
                loss = loss / torch.as_tensor(num_items_in_batch, device=loss.device)

        output = {"loss": loss, "logits": logits}
        if output_hidden_states:
            output["hidden_states"] = (x,)
        return output


# --- Model construction ------------------------------------------------------------------------
def build_model(vocab_size=27, hidden_size=128, num_layers=4, num_attention_heads=8,
                intermediate_size=None, max_len=512):
    """Build fresh weights after validating dimensions; None FFN width means 4 x hidden_size."""
    validate_architecture(
        hidden_size, num_layers, num_attention_heads, intermediate_size, max_len,
        rotary=True,
    )
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size

    logger.info(f"🚀 Initializing Causal Transformer from scratch "
                f"(hidden={hidden_size}, layers={num_layers}, heads={num_attention_heads}, "
                f"ffn={intermediate_size}, rotary positions)")
    return CausalAntibodyModel(CausalAntibodyConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max(max_len, 512),
    ))


# --- Registry hooks (shared names across AR model modules) ---------------------------------------
def load_tokenizer(source, position_type="relative", uses_special_tokens=False):
    """Return the custom causal tokenizer; source is unused for this fixed vocabulary."""
    from cdrseqgen.datasets.dataset_ar import causal_adapter
    return causal_adapter(position_type=position_type)


def load_training_model(source, *, vocab_size=None, hidden_size=128, num_layers=4,
                        num_attention_heads=8, intermediate_size=None, max_len=512, **_ignored):
    """Create fresh weights through build_model(); source is the registry's model name."""
    from cdrseqgen.datasets.dataset_ar import CausalTokenizer
    if vocab_size is None:
        vocab_size = len(CausalTokenizer().vocab)
    return build_model(
        vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers,
        num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
        max_len=max_len)


def load_pretrained_model(source):
    """Check the saved layout, then strictly restore the custom config and weights."""
    from cdrseqgen.methods.ar.backbones import validate_checkpoint_layout
    from cdrseqgen.methods.common.backbones import load_custom_pretrained
    tokenizer = load_tokenizer(source)
    validate_checkpoint_layout(source, tokenizer)
    return load_custom_pretrained(source, CausalAntibodyModel, CausalAntibodyConfig)


def validate_training_args(model_args, data_args):
    """Check custom architecture dimensions before the shared workflow creates a run."""
    validate_transformer_args(model_args, data_args, rotary=True)
