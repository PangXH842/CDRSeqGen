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


class TokenizerAdapter:
    """Uniform tokenizer interface so the Dataset never branches on model family.

    Wraps either the custom AntibodyTokenizer or any Hugging Face tokenizer and exposes a
    single contract: `encode` (never adds special tokens), `decode`, the id of every token
    kind the pipeline needs, plus two backbone conventions the encoding depends on:

      * uses_special_tokens -- wrap the assembled input as [cls] ... [end] (True for HF
        encoder LMs; False for the raw scratch tokenizer).
      * position_type -- "absolute" or "relative". Controls where variable-length antigen
        context is injected (see AntibodyInfillingDataset.__getitem__): absolute-position
        models (scratch, most BERT-style protein LMs) get it appended after the target so the
        target's absolute position stays stable; relative/rotary models (ESM-2) get it
        prepended. These come from the backbone registry, not guessed from the tokenizer.
    """

    def __init__(self, *, encode, decode, vocab_size, pad_id, mask_id, cls_id, end_id,
                 chain_sep_id, special_ids, uses_special_tokens, position_type):
        self.encode = encode
        self.decode = decode
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.cls_id = cls_id            # leading wrap token (None if unused)
        self.end_id = end_id            # trailing wrap token (None if unused)
        self.chain_sep_id = chain_sep_id
        self.special_ids = special_ids  # all non-amino-acid ids (for masking at inference)
        self.uses_special_tokens = uses_special_tokens
        self.position_type = position_type

    @classmethod
    def from_custom(cls, tok, position_type="absolute"):
        """Wrap the in-house AntibodyTokenizer (scratch model): no special-token wrapping."""
        return cls(
            encode=tok.encode,
            decode=tok.decode,
            vocab_size=len(tok.vocab),
            pad_id=tok.pad_id,
            mask_id=tok.mask_id,
            cls_id=None,
            end_id=None,
            chain_sep_id=tok.tok_to_id.get("<sep>", 3),
            special_ids={tok.tok_to_id[t] for t in ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>")},
            uses_special_tokens=False,
            position_type=position_type,
        )

    @classmethod
    def from_hf(cls, tok, position_type, uses_special_tokens=True):
        """Wrap any Hugging Face tokenizer (ESM-2, and future BERT/RoBERTa-style PLMs).

        Handles the ESM-vs-BERT separator difference generically: the trailing/chain
        separator is <eos> when the tokenizer has one (ESM/RoBERTa) and <sep> otherwise
        (BERT-style, where eos_token_id is None).
        """
        end_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
        cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        return cls(
            encode=lambda seq: tok.encode(seq, add_special_tokens=False),
            decode=lambda ids: tok.decode(ids).replace(" ", ""),
            vocab_size=tok.vocab_size,
            pad_id=tok.pad_token_id,
            mask_id=tok.mask_token_id,
            cls_id=cls_id,
            end_id=end_id,
            chain_sep_id=end_id,
            special_ids=set(tok.all_special_ids),
            uses_special_tokens=uses_special_tokens,
            position_type=position_type,
        )

class AntibodyInfillingDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=256, target_region="cdr3", include_light_chain=False, include_antigen=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_region = target_region
        self.include_light_chain = include_light_chain
        self.include_antigen = include_antigen

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

        # 2. Tokenize Heavy Chain Parts (adapter.encode never adds special tokens)
        t = self.tokenizer  # a TokenizerAdapter; see class docstring
        pt1_ids = t.encode(pt1_seq)
        pt2_ids = t.encode(pt2_seq)
        target_ids = t.encode(target_seq)

        # --- LIGHT CHAIN INJECTION ---
        # Prepend the paired light chain to pt1 (its length variance is small enough that this
        # doesn't destabilise the target's position even for absolute-position models).
        if self.include_light_chain and item.get('light') and item['light'].get('full_aa'):
            light_ids = t.encode(item['light']['full_aa'])
            pt1_ids = light_ids + [t.chain_sep_id] + pt1_ids

        # --- ANTIGEN INJECTION ---
        # Placement depends on the backbone's position_type (NOT the model family):
        #   relative (ESM-2 rotary)  -> prepend to pt1; the target's absolute index is irrelevant.
        #   absolute (scratch, BERT) -> append to pt2, keeping the target at a stable absolute
        #     position. Antigens vary enormously in length (up to ~2000+ residues), so prepending
        #     one ahead of an absolute-position model shifts the target unpredictably per example
        #     and the model fails to learn (empirically observed as a flat accuracy curve).
        if self.include_antigen and item.get('antigen') and item['antigen'].get('full_aa'):
            antigen_ids = t.encode(item['antigen']['full_aa'])
            if t.position_type == "relative":
                pt1_ids = antigen_ids + [t.chain_sep_id] + pt1_ids
            else:
                pt2_ids = pt2_ids + [t.chain_sep_id] + antigen_ids

        # --- LENGTH CAPPING ---
        # Cap whichever side carries the variable-length antigen context, always keeping the
        # heavy-chain residues immediately adjacent to the masked target and dropping the most
        # distal antigen context first.
        special_budget = 2 if t.uses_special_tokens else 0  # leading + trailing wrap tokens
        if t.position_type == "relative":
            # Antigen/light prepended to pt1; truncate its far (left) end.
            max_pt1_len = max(0, self.max_len - special_budget - len(target_ids) - len(pt2_ids))
            if len(pt1_ids) > max_pt1_len:
                pt1_ids = pt1_ids[len(pt1_ids) - max_pt1_len:]
        else:
            # Antigen appended to pt2; truncate its far (right) end.
            max_pt2_len = max(0, self.max_len - special_budget - len(pt1_ids) - len(target_ids))
            if len(pt2_ids) > max_pt2_len:
                pt2_ids = pt2_ids[:max_pt2_len]

        # 3. Construct Final Input and Find Mask Start
        mask_ids = [t.mask_id] * len(target_ids)
        if t.uses_special_tokens:
            input_ids = [t.cls_id] + pt1_ids + mask_ids + pt2_ids + [t.end_id]
            mask_start_idx = 1 + len(pt1_ids)  # 1 for the leading cls token
        else:
            input_ids = pt1_ids + mask_ids + pt2_ids
            mask_start_idx = len(pt1_ids)

        # 5. Create Labels (The "Safe" Way)
        labels = [-100] * len(input_ids)
        for i, t_id in enumerate(target_ids):
            labels[mask_start_idx + i] = t_id

        assert len(input_ids) == len(labels), f"Length Mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target_ids)
        }

def collate_fn(batch, pad_token_id=0):
    """
    Takes a list of items and pads them to the same length so they fit in a Tensor.

    pad_token_id defaults to 0, which is correct for AntibodyTokenizer's <pad> id, but
    NOT for ESM-2's tokenizer (whose <pad> id is 1, while 0 is <cls>) — callers using an
    ESM-2 tokenizer must pass its real pad_token_id explicitly (see train.py/predict.py).
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences to the longest one in this specific batch
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100 # -100 is ignored by Loss
    )

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids_padded != pad_token_id).long()

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask
    }

if __name__ == "__main__":
    # Test the dataset. The dataset expects a TokenizerAdapter, not a raw tokenizer.
    import functools
    adapter = TokenizerAdapter.from_custom(AntibodyTokenizer())
    dataset = AntibodyInfillingDataset("processed_paired_data.json", adapter)
    collate = functools.partial(collate_fn, pad_token_id=adapter.pad_id)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    # Peek at one batch
    for batch in dataloader:
        print("Input Shape:", batch["input_ids"].shape)
        print("First Input IDs:", batch["input_ids"][0])
        print("First Labels:", batch["labels"][0])
        break