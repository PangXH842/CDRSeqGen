"""Custom encoder-decoder Transformer for loop infilling.

Structure:
  Seq2SeqAntibodyConfig -> Seq2SeqAntibodyModel -> build_model()
  framework -> bidirectional rotary encoder -> decoder cross-attention
  decoder prefix -> causal rotary self-attention -> residue logits [B, S, V]

Contract:
  * Decoder inputs and labels are already offset by the dataset; loss does not shift.
  * Labels equal to -100 are ignored. num_items_in_batch normalizes the token-loss
    sum over the accumulation window; otherwise loss is a token mean.
  * encode() supports reusing framework states across candidate draws.
  * Optional hidden states expose the final decoder and encoder representations.
  * return_dict is accepted for HF compatibility; outputs are always dictionaries.
"""
import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from cdrseqgen.methods.common.transformer import (
    RotarySelfAttention as SelfAttention,
    feed_forward,
    init_transformer_weights,
    validate_architecture,
    validate_transformer_config,
)

logger = logging.getLogger(__name__)


class Seq2SeqAntibodyConfig(PretrainedConfig):
    """Persist resolved architecture dimensions under the existing checkpoint model_type."""

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


def _split_heads(x, n_head, head_dim):
    batch, seq, _ = x.shape
    return x.view(batch, seq, n_head, head_dim).transpose(1, 2)          # [B, H, S, D]


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


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.attn = SelfAttention(config, causal=False)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = feed_forward(config)

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
        self.mlp = feed_forward(config)

    def forward(self, x, encoder_hidden_states, key_padding_mask=None, encoder_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.cross(self.norm_cross(x), encoder_hidden_states,
                           encoder_padding_mask=encoder_padding_mask)
        return x + self.mlp(self.norm2(x))


class Seq2SeqAntibodyModel(PreTrainedModel):
    """Bidirectional encoder plus causal cross-attending decoder with aligned loss."""

    config_class = Seq2SeqAntibodyConfig
    # Trainer supplies the supervised-token count across the accumulation window.
    accepts_loss_kwargs = True

    def __init__(self, config):
        validate_transformer_config(config, rotary=True, encoder_decoder=True)
        super().__init__(config)
        # One embedding table for both streams: they are the same amino-acid vocabulary, and tying
        # them halves the parameter cost of the widest layer while keeping <bos>/<eos> consistent.
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.encoder_norm = nn.LayerNorm(config.hidden_size)
        self._build_decoder(config)
        self.post_init()

    def _build_decoder(self, config):
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _init_weights(self, module):
        init_transformer_weights(module)

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
    """Build fresh weights after validating dimensions; None FFN width means 4 x hidden_size."""
    validate_architecture(
        hidden_size, num_layers, num_attention_heads, intermediate_size, max_len,
        rotary=True,
    )
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    encoder_layers = num_layers if num_encoder_layers is None else num_encoder_layers
    decoder_layers = num_layers if num_decoder_layers is None else num_decoder_layers

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
