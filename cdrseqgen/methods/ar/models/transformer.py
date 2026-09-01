"""Custom causal decoder for from-scratch autoregressive infilling.

  * CausalAntibodyConfig -- PretrainedConfig subclass: the architecture spec HF saves and reloads
  * CausalAntibodyModel  -- architecture + forward; returns a {"loss", "logits"} DICT, matching the
                            masked model so samplers can share ``_logits_of``
  * build_model          -- master function: validates args, returns model object

Deliberately mirrors ``methods/mlm/models/transformer.py`` — same knobs, same dict return, same
``intermediate_size=None`` meaning 4 x hidden_size — so an architecture sweep behaves identically
across the two methods and results stay comparable. Two things differ, and both are forced:

  * **Attention is causal.** Position t may attend only to <= t. Without this the model sees the
    answer it is meant to predict and training collapses to copying.
  * **Positions are ROTARY, not learned-absolute.** The masked model uses learned absolute positions,
    and Findings §8 measured the cost: a prepended partner chain shifts every learned position, so a
    checkpoint is locked to the context regime it trained in (0.984 -> 0.261 when context is
    withheld) and needed `context_dropout` to survive both. The causal prompt's length varies by
    construction — partner chain and antigen are prepended, and capping trims a variable amount — so
    absolute positions would rebuild that fragility. Rotary encodes *relative* offsets, so the loop's
    relation to its flanks is unchanged by how much context precedes it.

Shifting: labels are supplied already aligned to inputs (``dataset_ar`` puts the native residue at
the position that predicts it), so the loss shifts internally — logits at t predict labels at t+1.
"""
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class CausalAntibodyConfig(PretrainedConfig):
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


def _rope_cache(seq_len, head_dim, device, dtype, theta):
    """(cos, sin) of shape [1, 1, seq_len, head_dim] for rotary embedding."""
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inverse)                 # [seq, head_dim/2]
    emb = torch.cat((angles, angles), dim=-1)                # [seq, head_dim]
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with rotary positions and a causal mask."""

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope_theta = config.rope_theta

    def forward(self, x, key_padding_mask=None):
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, H, S, D]

        cos, sin = _rope_cache(seq, self.head_dim, x.device, x.dtype, self.rope_theta)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        # SDPA cannot combine is_causal=True with an explicit mask, so when padding is present the
        # causal triangle is folded into that mask and is_causal stays off. Building it once matters:
        # the collator always supplies a padding mask, so this is the hot path.
        if key_padding_mask is None:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0)
        else:
            causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
            keep = key_padding_mask[:, None, None, :].bool() & causal[None, None]  # [B,1,S,S]
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.proj(out)


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.mlp(self.norm2(x))


class CausalAntibodyModel(PreTrainedModel):
    config_class = CausalAntibodyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([CausalBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, return_dict=True, **_kwargs):
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
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        output = {"loss": loss, "logits": logits}
        if output_hidden_states:
            output["hidden_states"] = (x,)
        return output


def build_model(vocab_size=27, hidden_size=128, num_layers=4, num_attention_heads=8,
                intermediate_size=None, max_len=512):
    """Build a fresh from-scratch causal transformer."""
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_attention_heads="
            f"{num_attention_heads} (attention splits the width across heads).")

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
