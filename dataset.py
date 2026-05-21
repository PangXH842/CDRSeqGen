import torch
from torch.utils.data import Dataset, DataLoader
import json

class AntibodyTokenizer:
    def __init__(self):
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
    def __init__(self, json_path, tokenizer, target_region="cdrh3", max_len=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_region = target_region

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        h = item['heavy']
        
        # 1. Define the regions based on target_region
        if self.target_region == "cdr1":
            pt1_seq = h['fwr1']
            target_seq = h['cdr1']
            pt2_seq = h['fwr2'] + h['cdr2'] + h['fwr3'] + h['cdr3'] + h['fwr4']
        elif self.target_region == "cdr2":
            pt1_seq = h['fwr1'] + h['cdr1'] + h['fwr2']
            target_seq = h['cdr2']
            pt2_seq = h['fwr3'] + h['cdr3'] + h['fwr4']
        else: # default to cdr3
            pt1_seq = h['fwr1'] + h['cdr1'] + h['fwr2'] + h['cdr2'] + h['fwr3']
            target_seq = h['cdr3']
            pt2_seq = h['fwr4']

        # 2. Identify Tokenizer Type
        is_esm = hasattr(self.tokenizer, "mask_token_id")

        # 3. Tokenize Parts
        if is_esm:
            # ESM (HuggingFace) Logic
            # We use add_special_tokens=False to prevent CLS/SEP in the middle
            pt1_ids = self.tokenizer.encode(pt1_seq, add_special_tokens=False)
            pt2_ids = self.tokenizer.encode(pt2_seq, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target_seq, add_special_tokens=False)
            
            mask_id = self.tokenizer.mask_token_id
            cls_id = self.tokenizer.cls_token_id
            eos_id = self.tokenizer.eos_token_id
            
            # Construct Input: <cls> pt1 <masks> pt2 <eos>
            mask_ids = [mask_id] * len(target_ids)
            input_ids = [cls_id] + pt1_ids + mask_ids + pt2_ids + [eos_id]
            
            # Start of the mask is after <cls> (index 1) and pt1
            mask_start_idx = 1 + len(pt1_ids)

        else:
            # Custom Scratch Tokenizer Logic
            pt1_ids = self.tokenizer.encode(pt1_seq)
            pt2_ids = self.tokenizer.encode(pt2_seq)
            target_ids = self.tokenizer.encode(target_seq)
            
            mask_id = self.tokenizer.mask_id
            
            # Construct Input: pt1 <masks> pt2
            mask_ids = [mask_id] * len(target_ids)
            input_ids = pt1_ids + mask_ids + pt2_ids
            
            # Start of mask is immediately after pt1
            mask_start_idx = len(pt1_ids)

        # 4. Create Labels (The "Safe" Way)
        # We create a list of -100 with the EXACT same length as input_ids
        labels = [-100] * len(input_ids)
        
        # We fill in the target amino acids only at the masked positions
        for i, t_id in enumerate(target_ids):
            labels[mask_start_idx + i] = t_id

        # 5. Final Safety Check
        assert len(input_ids) == len(labels), f"Length Mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target_ids)
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
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    # Peek at one batch
    for batch in dataloader:
        print("Input Shape:", batch["input_ids"].shape)
        print("First Input IDs:", batch["input_ids"][0])
        print("First Labels:", batch["labels"][0])
        break