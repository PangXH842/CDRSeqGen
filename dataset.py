import torch
from torch.utils.data import Dataset, DataLoader
import json

class AntibodyTokenizer:
    def __init__(self):
        # 20 standard Amino Acids + Special Tokens
        self.vocab = [
            "<pad>", "<unk>", "<cls>", "<sep>", "<mask>",
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
        ]
        self.tok_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_tok = {i: tok for i, tok in enumerate(self.vocab)}
        self.pad_id = self.tok_to_id["<pad>"]
        self.mask_id = self.tok_to_id["<mask>"]

    def encode(self, seq):
        return [self.tok_to_id.get(aa, self.tok_to_id["<unk>"]) for aa in seq]

    def decode(self, ids):
        return "".join([self.id_to_tok.get(i, "?") for i in ids])

class AntibodyInfillingDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        h = item['heavy']
        
        # 1. Define the regions
        fwr_context = h['fwr1'] + h['cdr1'] + h['fwr2'] + h['cdr2'] + h['fwr3']
        fwr4 = h['fwr4']
        h3 = h['cdr3']

        # 2. Tokenize
        context_ids = self.tokenizer.encode(fwr_context)
        fwr4_ids = self.tokenizer.encode(fwr4)
        h3_ids = self.tokenizer.encode(h3)

        # 3. Create Input: Context + [MASK] * len(H3) + FWR4
        # We use one MASK per amino acid so the model knows the target length
        mask_ids = [self.tokenizer.mask_id] * len(h3_ids)
        
        input_ids = context_ids + mask_ids + fwr4_ids
        
        # 4. Create Labels: 
        # In Masked LM, we usually set labels to -100 for non-masked tokens
        # PyTorch CrossEntropy ignores -100, so the model only learns the H3
        labels = ([-100] * len(context_ids)) + h3_ids + ([-100] * len(fwr4_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "h3_len": len(h3_ids)
        }

def collate_fn(batch):
    """
    Takes a list of items and pads them to the same length so they fit in a Tensor.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Pad sequences to the longest one in this specific batch
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0 # 0 is <pad>
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100 # -100 is ignored by Loss
    )
    # By setting everything except the CDR-H3 to -100, the model's loss will only be calculated based on 
    # how well it guesses the H3. It won't get "easy points" for guessing the Frameworks it already sees.
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_padded != 0).long()

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask
    }

if __name__ == "__main__":
    # Test the dataset
    tokenizer = AntibodyTokenizer()
    dataset = AntibodyInfillingDataset("processed_paired_data.json", tokenizer)
    
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # Peek at one batch
    for batch in dataloader:
        print("Input Shape:", batch["input_ids"].shape)
        print("First Input IDs:", batch["input_ids"][0])
        print("First Labels:", batch["labels"][0])
        break