"""Building blocks shared by the current custom Transformers.

Common functions:
  * validate_*               -- check architecture dimensions in CLI, builder and model paths
  * init_transformer_weights -- initialize Linear/Embedding weights and linear biases
  * feed_forward             -- construct a Linear -> GELU -> Linear FFN

Attention and positional encoding:
  * RotarySelfAttention -- causal or bidirectional self-attention with rotary positions
  * _rope_cache         -- compute rotary cosine/sine tensors
  * _rotate_half,
    _apply_rope         -- apply rotary transformations to queries and keys

Method-owned components:
  * Learned absolute positions -- MLM's position table, defined in its model module
  * Model layout, normalization, cross-attention and label/loss alignment
"""
import torch
from torch import nn


def validate_positive_integer(name, value):
    """Reject zero, negative, fractional and boolean architecture counts."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        depth = "; at least one block is required" if "layers" in name else ""
        raise ValueError(f"{name} must be a positive integer{depth}, got {value!r}")


def validate_attention_dimensions(hidden_size, num_attention_heads, *, rotary=False):
    """Check head divisibility and the even head width required by rotary positions."""
    validate_positive_integer("hidden_size", hidden_size)
    validate_positive_integer("num_attention_heads", num_attention_heads)
    if hidden_size % num_attention_heads:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}"
        )
    if rotary and (hidden_size // num_attention_heads) % 2:
        raise ValueError(
            "rotary attention requires an even head dimension (even head width): "
            f"hidden_size={hidden_size} / num_attention_heads={num_attention_heads}"
        )


def validate_architecture(hidden_size, num_layers, num_attention_heads,
                          intermediate_size, max_len, *, rotary=False):
    """Validate builder/CLI settings; intermediate_size=None requests the 4× width default."""
    validate_attention_dimensions(hidden_size, num_attention_heads, rotary=rotary)
    validate_positive_integer("num_layers", num_layers)
    validate_positive_integer("max_len", max_len)
    if intermediate_size is not None:
        validate_positive_integer("intermediate_size", intermediate_size)


def validate_transformer_config(config, *, rotary=False, encoder_decoder=False):
    """Validate resolved dimensions before a custom model allocates parameters."""
    validate_architecture(
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        config.intermediate_size, config.max_position_embeddings, rotary=rotary,
    )
    validate_positive_integer("vocab_size", config.vocab_size)
    # Saved configs describe a concrete FFN size; only builders accept None.
    validate_positive_integer("intermediate_size", config.intermediate_size)
    if encoder_decoder:
        validate_positive_integer("num_decoder_layers", config.num_decoder_layers)


def validate_transformer_args(model_args, data_args, *, rotary=False):
    """Validate CLI architecture settings without constructing or downloading a model."""
    validate_architecture(
        model_args.hidden_size, model_args.num_layers, model_args.num_attention_heads,
        model_args.intermediate_size, data_args.max_len, rotary=rotary,
    )
    for name in ("num_encoder_layers", "num_decoder_layers"):
        value = getattr(model_args, name, None)
        if value is not None:
            validate_positive_integer(name, value)


def init_transformer_weights(module):
    """Apply the existing std-0.02 linear/embedding initialization in traversal order."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def feed_forward(config):
    """Return the linear → GELU → linear FFN used by the AR and S2S blocks."""
    return nn.Sequential(
        nn.Linear(config.hidden_size, config.intermediate_size),
        nn.GELU(),
        nn.Linear(config.intermediate_size, config.hidden_size),
    )


def _rope_cache(seq_len, head_dim, device, dtype, theta):
    """Return cosine/sine tensors [1, 1, sequence, head_dim], computed in float32."""
    inverse = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, inverse)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x, cos, sin):
    return x * cos + _rotate_half(x) * sin


class RotarySelfAttention(nn.Module):
    """Apply rotary self-attention, optionally masking future positions.

    key_padding_mask uses True/1 for real tokens. For a padded causal input,
    padding and the causal triangle form one boolean SDPA mask. qkv/proj remain
    direct children so existing checkpoints retain their parameter names.
    """

    def __init__(self, config, causal):
        super().__init__()
        validate_attention_dimensions(
            config.hidden_size, config.num_attention_heads, rotary=True
        )
        self.causal = causal
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope_theta = config.rope_theta

    def forward(self, x, key_padding_mask=None):
        batch, seq, _ = x.shape
        qkv = self.qkv(x).view(batch, seq, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        cos, sin = _rope_cache(seq, self.head_dim, x.device, x.dtype, self.rope_theta)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)

        if key_padding_mask is None:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal, dropout_p=0.0)
        else:
            keep = key_padding_mask[:, None, None, :].bool()
            if self.causal:
                causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
                keep = keep & causal[None, None]
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.proj(out)
