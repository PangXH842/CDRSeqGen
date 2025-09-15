import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

# --- Vocabulary (remains the same) ---
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# SPECIAL_TOKENS = {"<pad>": 0, "<cls>": 1, "<sep>": 2, "<mask>": 3}
SPECIAL_TOKENS = {
    "<pad>": 0,  # Padding token
    "X": 1,      # Mask token for unknown amino acids
}

class Vocabulary:
    def __init__(self):
        self.token_to_idx = {**SPECIAL_TOKENS}
        for i, aa in enumerate(AMINO_ACIDS):
            self.token_to_idx[aa] = i + len(SPECIAL_TOKENS)
        
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}

    def __len__(self):
        return len(self.token_to_idx)

    def tokenize(self, sequence: str) -> torch.Tensor:
        return torch.tensor([self.token_to_idx[token] for token in sequence], dtype=torch.long)

# Instantiate a global vocabulary object
vocab = Vocabulary()

# --- The Simplified Dataset Class ---

class InfillingDataset(Dataset):
    def __init__(self, data_file: str, split: str = 'train'):
        super().__init__()
        data = torch.load(data_file)
        self.data = data.get(split)

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        
        # The input to the model is the heavy chain with the CDR-H3 masked with 'X'
        input_tokens = vocab.tokenize(item['masked_heavy_chain'])
        
        # The target label is the full, unmasked heavy chain
        target_tokens = vocab.tokenize(item['full_heavy_chain'])
        
        return {'input_tokens': input_tokens, 'target_tokens': target_tokens}

# --- The Simplified Collate Function ---

def infilling_collate_fn(batch: list) -> dict:
    input_tokens = [item['input_tokens'] for item in batch]
    target_tokens = [item['target_tokens'] for item in batch]
    
    # Pad all sequences to the same length in the batch
    inputs_padded = pad_sequence(input_tokens, batch_first=True, padding_value=vocab.token_to_idx["<pad>"])
    targets_padded = pad_sequence(target_tokens, batch_first=True, padding_value=vocab.token_to_idx["<pad>"])
    
    # The padding mask tells the model which tokens to ignore (the padding)
    padding_mask = (inputs_padded == vocab.token_to_idx["<pad>"])
    
    return {
        'input_tokens': inputs_padded,
        'target_tokens': targets_padded,
        'padding_mask': padding_mask,
    }