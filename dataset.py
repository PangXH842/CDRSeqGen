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
        
        # Dynamically find where the target is
        if self.target_region == "cdrh1":
            fwr_pt1 = h['fwr1']
            fwr_target = h['cdrh1']
            fwr_pt2 = h['fwr2'] + h['cdrh2'] + h['fwr3'] + h['cdrh3'] + h['fwr4']
        elif self.target_region == "cdrh2":
            fwr_pt1 = h['fwr1'] + h['cdr1'] + h['fwr2']
            fwr_target = h['cdrh2']
            fwr_pt2 = h['fwr3'] + h['cdrh3'] + h['fwr4']
        else: # default to cdrh3
            fwr_pt1 = h['fwr1'] + h['cdr1'] + h['fwr2'] + h['cdr2'] + h['fwr3']
            fwr_target = h['cdrh3']
            fwr_pt2 = h['fwr4']

        # Tokenize 
        pt1_ids = self.tokenizer.encode(fwr_pt1)
        pt2_ids = self.tokenizer.encode(fwr_pt2)
        target_ids = self.tokenizer.encode(fwr_target)

        # 3. Create Input: We use one MASK per amino acid so the model knows the target length
        mask_ids = [self.tokenizer.mask_id] * len(target_ids)
        
        input_ids = pt1_ids + mask_ids + pt2_ids
        
        # 4. Create Labels: 
        # In Masked LM, we usually set labels to -100 for non-masked tokens
        # PyTorch CrossEntropy ignores -100, so the model only learns the target, eg. H3
        labels = ([-100] * len(pt1_ids)) + target_ids + ([-100] * len(pt2_ids))

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