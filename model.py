import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, EsmForMaskedLM, EsmConfig

# --- APPROACH 1: Custom Transformer from Scratch ---

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
        # 1. Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 2. Transformer (We ignore attention_mask for this simple baseline, but you can add it)
        x = self.transformer(x)
        
        # 3. Output Logits
        logits = self.lm_head(x)
        
        # 4. Standard HuggingFace output format
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # ignore_index=-100 is default
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

# --- THE FACTORY FUNCTION ---

def get_model(model_name_or_path, vocab_size=25):
    """
    Returns either a custom model or an ESM model based on the name.
    """
    if model_name_or_path == "scratch":
        print("🚀 Initializing Custom Transformer from scratch...")
        config = CustomAntibodyConfig(vocab_size=vocab_size)
        return CustomAntibodyModel(config)
    
    elif "/esm2" in model_name_or_path:
        print(f"🧬 Loading Pre-trained ESM-2 Model: {model_name_or_path}")
        # ESM-2 already comes with its own vocab and configuration
        return EsmForMaskedLM.from_pretrained(model_name_or_path)
    
    else:
        raise ValueError("Unknown model choice. Choose 'scratch' or an ESM-2 path.")