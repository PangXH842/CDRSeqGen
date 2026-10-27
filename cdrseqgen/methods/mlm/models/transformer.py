"""Custom bidirectional encoder for from-scratch masked-LM training.

  * CustomAntibodyConfig    -- PretrainedConfig subclass: the architecture spec HF saves and reloads
  * CustomAntibodyModel     -- architecture + forward; returns a {"loss", "logits"} DICT, which is
                               why the shared ``logits_of`` accepts a dict as well as an object
  * build_model             -- master function: validates args, returns model object

Note:
  * Positions are learned and absolute, so the dataset appends antigen context after the target
    so the loop keeps a stable index
  * Logs via the root logger, configured by the corresponding entrypoint
    (train.py, generate.py, or evaluate.py)
  * CustomAntibodyModel.from_pretrained() returns RANDOMLY INITIALIZED weights with no missing-key
    warning. Must instead use _load_custom_pretrained from backbones.py, which loads the state dict
    explicitly with strict=True
  * All four architecture knobs (hidden, layer, head, ffn) come from the config (ModelArguments),
    so an architecture sweep can move any of them without editing this file. Model dispatch lives
    in backbones.py; this is just the scratch builder.
  * max_position_embeddings is sized from max_len (never below the 512 default) so a longer max_len
    can't overflow the position-embedding table — the model has one learned vector per slot.
  * intermediate_size=None means 4 x hidden_size, the standard transformer FFN (feed-forward
    network) ratio. Until 2026-07-27 it was pinned at 512 while hidden_size was a free parameter,
    so widening the model silently shrank the ratio (256 -> 2x, 512 -> 1x) and an "architecture
    sweep" would have confounded width with FFN ratio. Pass a number only to break it on purpose.
"""

import logging

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

logger = logging.getLogger(__name__)

class CustomAntibodyConfig(PretrainedConfig):
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
    config_class = CustomAntibodyConfig

    def __init__(self, config):
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
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

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
    """Build a fresh custom-from-scratch transformer"""
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_attention_heads="
            f"{num_attention_heads} (nn.MultiheadAttention splits the width across heads)."
        )

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
