import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings.
    Transformers do not inherently know the order of the sequence, so this is crucial.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as a buffer, not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embed_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class InfillingTransformer(nn.Module):
    """
    An Encoder-Only Transformer model for the CDR-H3 infilling task.
    This architecture is similar to BERT.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 512,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 hidden_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # The core of the model is a stack of Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # The final output layer projects the hidden states to vocabulary size
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize weights for better training stability
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self,
                src_tokens: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass.

        Args:
            src_tokens (Tensor): The masked heavy chain sequences. Shape: [batch_size, seq_len]
            padding_mask (Tensor): Mask for padding tokens. Shape: [batch_size, seq_len]

        Returns:
            Tensor: Raw output logits. Shape: [batch_size, seq_len, vocab_size]
        """
        # 1. Embed inputs
        embed = self.embedding(src_tokens) * math.sqrt(self.embed_dim)
        embed = self.pos_encoder(embed.transpose(0, 1)).transpose(0, 1)

        # 2. Pass through the Transformer Encoder
        # The bidirectional self-attention allows every token to see every other token
        # to understand the full context before making a prediction.
        output = self.transformer_encoder(
            src=embed,
            src_key_padding_mask=padding_mask
        )

        # 3. Project to vocabulary size
        logits = self.output_layer(output)

        return logits

# class ConditionalTransformer(nn.Module):
#     """
#     A conditional sequence-to-sequence Transformer model.

#     This model is designed for the task of antigen-specific CDR-H3 generation.
#     - The Encoder processes the antigen sequence to create a context vector.
#     - The Decoder generates the CDR-H3 sequence, conditioned on the antibody framework
#       and the antigen context from the encoder.
#     """
#     def __init__(self,
#                  vocab_size: int,
#                  embed_dim: int = 512,
#                  num_layers: int = 6,
#                  num_heads: int = 8,
#                  hidden_dim: int = 2048,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.embed_dim = embed_dim

#         # --- Shared Components ---
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
#         # --- Encoder ---
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             batch_first=True  # Important: This makes the API expect (batch, seq, feature)
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         # --- Decoder ---
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             dim_feedforward=hidden_dim,
#             dropout=dropout,
#             batch_first=True  # Important
#         )
#         self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#         # --- Final Output Layer ---
#         self.output_layer = nn.Linear(embed_dim, vocab_size)

#         self._init_weights()

#     def _init_weights(self) -> None:
#         # Initialize weights for better training stability
#         initrange = 0.1
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.output_layer.bias.data.zero_()
#         self.output_layer.weight.data.uniform_(-initrange, initrange)

#     def forward(self,
#                 antigen_tokens: torch.Tensor,
#                 antibody_tokens: torch.Tensor,
#                 src_padding_mask: torch.Tensor,
#                 tgt_padding_mask: torch.Tensor,
#                 tgt_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Performs a forward pass of the model during training.

#         Args:
#             antigen_tokens (Tensor): The antigen sequence tokens. Shape: [batch_size, antigen_len]
#             antibody_tokens (Tensor): The antibody framework + CDR sequence tokens (for teacher forcing).
#                                       Shape: [batch_size, antibody_len]
#             src_padding_mask (Tensor): Mask for padding in the antigen sequence. Shape: [batch_size, antigen_len]
#             tgt_padding_mask (Tensor): Mask for padding in the antibody sequence. Shape: [batch_size, antibody_len]
#             tgt_mask (Tensor): A causal mask to prevent the decoder from seeing future tokens.
#                                Shape: [antibody_len, antibody_len]

#         Returns:
#             Tensor: The raw output logits from the model. Shape: [batch_size, antibody_len, vocab_size]
#         """
#         # 1. Embed inputs
#         antigen_embed = self.embedding(antigen_tokens) * math.sqrt(self.embed_dim)
#         antigen_embed = self.pos_encoder(antigen_embed.transpose(0, 1)).transpose(0, 1) # pos_encoder expects (seq, batch, feature)

#         antibody_embed = self.embedding(antibody_tokens) * math.sqrt(self.embed_dim)
#         antibody_embed = self.pos_encoder(antibody_embed.transpose(0, 1)).transpose(0, 1)

#         # 2. Encoder Pass: Process the antigen to create the context memory
#         # The encoder's output (`memory`) is its "understanding" of the antigen.
#         memory = self.encoder(
#             src=antigen_embed,
#             src_key_padding_mask=src_padding_mask
#         )

#         # 3. Decoder Pass: Generate the CDR based on the framework and antigen memory
#         # The decoder attends to both the antibody sequence so far (`antibody_embed`)
#         # and the antigen context (`memory`).
#         output = self.decoder(
#             tgt=antibody_embed,
#             memory=memory,
#             tgt_mask=tgt_mask,
#             tgt_key_padding_mask=tgt_padding_mask,
#             memory_key_padding_mask=src_padding_mask
#         )

#         # 4. Final Projection: Convert the decoder's output to vocabulary logits
#         logits = self.output_layer(output)

#         return logits

# This block allows you to run the file directly to test the model definition
if __name__ == '__main__':
    # --- Test the model with dummy data ---
    VOCAB_SIZE = 25  # ~20 amino acids + special tokens like <PAD>, <MASK>, etc.
    EMBED_DIM = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    
    # Create a model instance
    model = InfillingTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS
    )
    print(f"Model created with ~{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # Create dummy input tensors
    batch_size = 4
    antigen_len = 150
    antibody_len = 130 # Framework + CDR

    antigen_input = torch.randint(0, VOCAB_SIZE, (batch_size, antigen_len))
    antibody_input = torch.randint(0, VOCAB_SIZE, (batch_size, antibody_len))
    
    # Create necessary masks (a real data loader would do this)
    src_pad_mask = (antigen_input == 0) # Assume 0 is the padding token
    tgt_pad_mask = (antibody_input == 0)
    
    # Create the causal mask for the decoder
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(antibody_len)

    # Run a forward pass
    model.eval()
    with torch.no_grad():
        logits = model(
            antigen_tokens=antigen_input,
            antibody_tokens=antibody_input,
            src_padding_mask=src_pad_mask,
            tgt_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask
        )

    print(f"Input antigen shape: {antigen_input.shape}")
    print(f"Input antibody shape: {antibody_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("Model test successful!")