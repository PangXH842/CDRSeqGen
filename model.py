import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# --- Custom Transformer from Scratch ---

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

    def forward(self, input_ids, attention_mask=None, labels=None):
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
            
        return {"loss": loss, "logits": logits}

# --- THE FACTORY FUNCTION ---

def build_scratch_model(vocab_size=25, hidden_size=128, num_layers=4, max_len=512):
    """Build a fresh custom-from-scratch transformer.

    hidden_size/num_layers come straight from the config (ModelArguments); max_position_embeddings
    is sized from max_len (never below the 512 default) so a longer max_len can't overflow the
    position-embedding table. Model dispatch lives in backbones.py; this is just the scratch builder.
    """
    print("🚀 Initializing Custom Transformer from scratch...")
    config = CustomAntibodyConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        max_position_embeddings=max(max_len, 512),
    )
    return CustomAntibodyModel(config)