"""Custom causal Transformer for autoregressive loop infilling.

Structure:
  CausalAntibodyConfig -> CausalAntibodyModel -> build_model()
  token embedding -> causal rotary blocks -> norm -> logits [B, S, V]

Contract:
  * Inputs and labels share positions; loss pairs logits[:, :-1] with labels[:, 1:].
  * Prompt/padding labels are -100. A supplied num_items_in_batch normalizes the
    token-loss sum over the accumulation window; otherwise loss is a token mean.
  * Optional hidden_states contains the final normalized representation.
  * return_dict is accepted for HF compatibility; outputs are always dictionaries.
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
    validate_transformer_config,
)

logger = logging.getLogger(__name__)


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
        x = self.embedding(input_ids)
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Next-token prediction: the logit at t is scored against the label at t+1. The dataset
            # already places -100 across the prompt, so only the answer and its <eos> contribute.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Normalize across the full accumulation window when Trainer supplies its count. A
            # model that takes **kwargs makes Trainer stop dividing by gradient_accumulation_steps,
            # so a plain microbatch mean here would scale the gradient by that factor (measured:
            # 2.02x at two steps). Direct calls and evaluation retain ordinary token-mean loss.
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1),
                reduction="mean" if num_items_in_batch is None else "sum")
            if num_items_in_batch is not None:
                loss = loss / torch.as_tensor(num_items_in_batch, device=loss.device)

        output = {"loss": loss, "logits": logits}
        if output_hidden_states:
            output["hidden_states"] = (x,)
        return output


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
