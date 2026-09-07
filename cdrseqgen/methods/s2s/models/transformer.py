"""Custom encoder-decoder for from-scratch span infilling.

  * Seq2SeqAntibodyConfig -- PretrainedConfig subclass: the architecture spec HF saves and reloads
  * Seq2SeqAntibodyModel  -- architecture + forward; returns a {"loss", "logits"} DICT, matching the
                             masked and causal models so samplers can share ``logits_of``
  * build_model           -- master function: validates args, returns model object

Deliberately mirrors ``methods/ar/models/transformer.py`` — same knobs, same dict return, same
``intermediate_size=None`` meaning 4 x hidden_size, same rotary positions — so an architecture sweep
behaves identically across methods. Three things differ, and all three are forced:

  * **The encoder is bidirectional.** Its self-attention has no causal mask, so FR1 and FR4 inform
    each other before a residue is emitted. This is the entire architectural claim of the method.

  * **The decoder has cross-attention.** Each decoder block attends to its own prefix (causal) and
    then to the full encoder output. Rotary is applied to SELF-attention only: query and key live in
    the same position space there, while a cross-attention query (loop position 3) and key
    (framework position 140) do not, and rotating across two unrelated coordinate systems encodes a
    relative offset that means nothing.

  * **The loss does NOT shift.** ``dataset_s2s.build_decoder_pair`` hands over inputs and labels
    already offset by one, so logits at t are scored against labels at t directly. The causal model
    shifts internally because ``dataset_ar`` supplies aligned labels instead. Shifting in both places
    trains perfectly happily against the wrong target and yields a checkpoint that looks merely bad,
    which is why the convention is stated in both files and asserted in ``test_s2s_model``.

**Capacity is NOT matched to an `ar` run with the same ``num_layers``.** ``num_layers=4`` builds four
encoder blocks AND four decoder blocks, and the decoder blocks additionally carry cross-attention, so
the model is more than twice the size of the causal one at the same setting. That is deliberate:
silently halving the depth to fake a match would itself be an un-audited hyperparameter, and
Findings §9 records what one of those cost. ``build_model`` logs the parameter count so the number
is visible in every ``train.log``, and ``num_encoder_layers`` / ``num_decoder_layers`` are separate
knobs so the ``arch`` sweep can resolve capacity-versus-architecture empirically rather than by
assertion. Any table putting `s2s` beside `ar` must quote both parameter counts.
"""
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)


class Seq2SeqAntibodyConfig(PretrainedConfig):
    model_type = "seq2seq_antibody_transformer"

    def __init__(self, vocab_size=27, hidden_size=128, num_hidden_layers=4,
                 num_decoder_layers=4, num_attention_heads=8, intermediate_size=512,
                 max_position_embeddings=512, rope_theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers          # encoder depth
        self.num_decoder_layers = num_decoder_layers
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


def _split_heads(x, n_head, head_dim):
    batch, seq, _ = x.shape
    return x.view(batch, seq, n_head, head_dim).transpose(1, 2)          # [B, H, S, D]


class SelfAttention(nn.Module):
    """Multi-head self-attention with rotary positions, causal or bidirectional."""

    def __init__(self, config, causal):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.causal = causal
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        if self.head_dim % 2:
            raise ValueError("rotary attention requires an even head dimension "
                             "(hidden_size / num_attention_heads)")
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
        # causal triangle is folded into that mask and is_causal stays off.
        if key_padding_mask is None:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=self.causal, dropout_p=0.0)
        else:
            keep = key_padding_mask[:, None, None, :].bool()              # [B,1,1,S]
            if self.causal:
                causal = torch.ones(seq, seq, device=x.device, dtype=torch.bool).tril()
                keep = keep & causal[None, None]                          # [B,1,S,S]
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Decoder-to-encoder attention. No rotary: the two streams have unrelated position spaces."""

    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.kv = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, encoder_hidden_states, encoder_padding_mask=None):
        batch, seq, _ = x.shape
        q = _split_heads(self.q(x), self.n_head, self.head_dim)
        kv = self.kv(encoder_hidden_states)
        k, v = kv.chunk(2, dim=-1)
        k = _split_heads(k, self.n_head, self.head_dim)
        v = _split_heads(v, self.n_head, self.head_dim)

        attn_mask = None
        if encoder_padding_mask is not None:
            attn_mask = encoder_padding_mask[:, None, None, :].bool()     # [B,1,1,S_enc]
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        return self.proj(out)


def _feed_forward(config):
    return nn.Sequential(
        nn.Linear(config.hidden_size, config.intermediate_size),
        nn.GELU(),
        nn.Linear(config.intermediate_size, config.hidden_size),
    )


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = SelfAttention(config, causal=False)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = _feed_forward(config)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        return x + self.mlp(self.norm2(x))


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = SelfAttention(config, causal=True)
        self.norm_cross = nn.LayerNorm(config.hidden_size)
        self.cross = CrossAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = _feed_forward(config)

    def forward(self, x, encoder_hidden_states, key_padding_mask=None, encoder_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.cross(self.norm_cross(x), encoder_hidden_states,
                           encoder_padding_mask=encoder_padding_mask)
        return x + self.mlp(self.norm2(x))


class Seq2SeqAntibodyModel(PreTrainedModel):
    config_class = Seq2SeqAntibodyConfig
    # Trainer supplies the supervised-token count across the accumulation window.
    accepts_loss_kwargs = True

    def __init__(self, config):
        super().__init__(config)
        # One embedding table for both streams: they are the same amino-acid vocabulary, and tying
        # them halves the parameter cost of the widest layer while keeping <bos>/<eos> consistent.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.encoder_norm = nn.LayerNorm(config.hidden_size)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def encode(self, input_ids, attention_mask=None):
        """Run the encoder once. Sampling reuses one call across every candidate for a framework."""
        x = self.embedding(input_ids)
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.encoder_norm(x)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, encoder_hidden_states=None,
                output_hidden_states=False, return_dict=True, num_items_in_batch=None, **_kwargs):
        if encoder_hidden_states is None:
            if input_ids is None:
                raise ValueError("seq2seq forward needs input_ids or encoder_hidden_states")
            encoder_hidden_states = self.encode(input_ids, attention_mask=attention_mask)
        if decoder_input_ids is None:
            raise ValueError("seq2seq forward needs decoder_input_ids; this model does not build "
                             "them from labels, so a caller cannot accidentally shift twice")

        x = self.embedding(decoder_input_ids)
        key_padding_mask = (decoder_attention_mask.bool()
                            if decoder_attention_mask is not None else None)
        encoder_padding_mask = attention_mask.bool() if attention_mask is not None else None
        for block in self.decoder_blocks:
            x = block(x, encoder_hidden_states, key_padding_mask=key_padding_mask,
                      encoder_padding_mask=encoder_padding_mask)
        x = self.decoder_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # NO shift: dataset_s2s already offset the pair, so the logit at t is scored against the
            # label at t. Padding is -100 and drops out.
            # Normalize across the full accumulation window when Trainer supplies its count.
            # Summing microbatch means would overweight short loops and scale gradients with
            # gradient_accumulation_steps. Direct calls retain ordinary token-mean loss.
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), labels.reshape(-1),
                reduction="mean" if num_items_in_batch is None else "sum")
            if num_items_in_batch is not None:
                loss = loss / torch.as_tensor(num_items_in_batch, device=loss.device)

        output = {"loss": loss, "logits": logits}
        if output_hidden_states:
            output["hidden_states"] = (x,)
            output["encoder_hidden_states"] = (encoder_hidden_states,)
        return output


def build_model(vocab_size=27, hidden_size=128, num_layers=4, num_attention_heads=8,
                intermediate_size=None, max_len=512, num_encoder_layers=None,
                num_decoder_layers=None):
    """Build a fresh from-scratch encoder-decoder transformer."""
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_attention_heads="
            f"{num_attention_heads} (attention splits the width across heads).")
    encoder_layers = num_layers if num_encoder_layers is None else num_encoder_layers
    decoder_layers = num_layers if num_decoder_layers is None else num_decoder_layers
    if encoder_layers < 1 or decoder_layers < 1:
        raise ValueError("an encoder-decoder needs at least one block on each side; got "
                         f"encoder={encoder_layers}, decoder={decoder_layers}")

    model = Seq2SeqAntibodyModel(Seq2SeqAntibodyConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=encoder_layers,
        num_decoder_layers=decoder_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max(max_len, 512),
    ))
    # Logged, not asserted: this is NOT capacity-matched to an `ar` model of the same num_layers,
    # and the number belongs in train.log so a comparison table can quote it.
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"🚀 Initializing Seq2Seq Transformer from scratch "
                f"(hidden={hidden_size}, enc_layers={encoder_layers}, dec_layers={decoder_layers}, "
                f"heads={num_attention_heads}, ffn={intermediate_size}, rotary positions) "
                f"— {params:,} parameters")
    return model
