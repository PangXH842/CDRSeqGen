"""Dataset class script, preprocessed jsons -> torch tensors for training (MLM infilling).

  * AntibodyTokenizer        -- custom 25-token vocab (20 amino acids + 5 special)
  * TokenizerAdapter         -- uniform tokenizer interface (custom, HF)
  * assemble_masked_input    -- the [cls] pt1 [mask x L] pt2 [end] layout, shared with predict.py
  * AntibodyInfillingDataset -- one entry -> (input_ids, labels, target_len) with one CDR masked
  * collate_fn               -- pad a list of items into a batch (+ attention mask)

Input schema:
    heavy / light            -- {fwr1..fwr4, cdr1..cdr3, full_aa, imgt_numbering}
    antigen (SAbDab2 only)   -- {full_aa, chains, type}
    split (optional)         -- "train"/"test", honored by get_datasets() in train.py

Notes:
  * Layout is: [cls] pt1 [mask x L] pt2 [end], where pt1/pt2 are the framework flanks either side of
    the target loop; the wrap tokens are dropped for the scratch tokenizer (uses_special_tokens).
  * Loss is scored on the loop alone: labels are -100 everywhere except the masked span.
  * The mask span is exactly as long as the native loop, so the model is TOLD the target length and
    never has to predict it -- recovery numbers should be read with that in mind.
  * Where partner-chain / antigen context is injected depends on the backbone's position_type, not
    its model family -- see TokenizerAdapter and __getitem__.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import json

class AntibodyTokenizer:
    """Token vocab for scratch model: 20 amino acids + 5 special tokens"""

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
        """Residue string -> list of ids; any non-canonical residue becomes <unk>"""
        return [self.tok_to_id.get(aa, self.tok_to_id["<unk>"]) for aa in seq]

    def decode(self, ids):
        """List of ids -> residue string; ids outside the vocab render as '?'"""
        return "".join([self.id_to_tok.get(i, "?") for i in ids])


class TokenizerAdapter:
    """Uniform tokenizer interface for custom and hf sources"""

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
    def from_hf(cls, tok, position_type, uses_special_tokens=True, space_residues=False):
        """Wrap any Hugging Face tokenizer (ESM-2, and BERT/RoBERTa-style PLMs)

        Cases:
          Wrap tokens (auto-detected; end_id doubles as chain_sep_id)
          * ESM/RoBERTa: cls_id = <cls>; end_id = <eos>
          * BERT-style:  cls_id = <cls>; end_id = <sep>
          Residue spacing (NOT detected — caller passes space_residues)
          * ESM/RoBERTa: "MVEQ"
          * BERT-style:  "M E V Q"; bare "MVEQ" -> a single <unk>
        """
        cls_id = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
        end_id = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id

        def _encode(seq):
            return tok.encode(" ".join(seq) if space_residues else seq, add_special_tokens=False)

        return cls(
            encode=_encode,
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

def inject_context(tokenizer, pt1_ids, pt2_ids, n_target, *,
                   partner_seq=None, antigen_seq=None, max_len=256):
    """Add partner-chain / antigen context around the target, then cap to max_len.

    Returns the (pt1_ids, pt2_ids) to hand to assemble_masked_input. `partner_seq`/`antigen_seq` are
    raw residue strings (None = that context is off).

    Shared deliberately, for the same reason as assemble_masked_input: AntibodyInfillingDataset
    builds TRAINING inputs with this and predict.py builds INFERENCE inputs with it. A checkpoint
    trained with context but fed without it does not crash — it silently degrades (measured: an
    `_LC` H1 model drops 0.986 -> 0.260 recovery when the light chain is missing), so the two paths
    must construct context identically or inference is quietly wrong.

    Placement rules, both keyed on the backbone's position_type rather than the model family:
      * partner chain -> always PREPENDED to pt1. Its length varies little (light chains ~96-118
        residues), so it doesn't destabilise the target's absolute index.
      * antigen -> PREPENDED to pt1 for relative/rotary models (ESM-2), where the target's absolute
        index is irrelevant; APPENDED to pt2 for absolute-position models (scratch, BERT-style), so
        the target keeps a stable index. Antigens run to ~2,300 residues, so prepending one ahead of
        an absolute-position model shifts the target unpredictably per example and it fails to learn
        (empirically: a flat accuracy curve).
      * capping trims whichever side carries the antigen, from its far end, so the most distal
        context is dropped first and the residues adjacent to the loop always survive.
    """
    t = tokenizer
    if partner_seq:
        pt1_ids = t.encode(partner_seq) + [t.chain_sep_id] + pt1_ids

    if antigen_seq:
        antigen_ids = t.encode(antigen_seq)
        if t.position_type == "relative":
            pt1_ids = antigen_ids + [t.chain_sep_id] + pt1_ids
        else:  # absolute
            pt2_ids = pt2_ids + [t.chain_sep_id] + antigen_ids

    special_budget = 2 if t.uses_special_tokens else 0  # leading + trailing wrap tokens
    if t.position_type == "relative":
        max_pt1_len = max(0, max_len - special_budget - n_target - len(pt2_ids))
        if len(pt1_ids) > max_pt1_len:
            pt1_ids = pt1_ids[len(pt1_ids) - max_pt1_len:]
    else:
        max_pt2_len = max(0, max_len - special_budget - len(pt1_ids) - n_target)
        if len(pt2_ids) > max_pt2_len:
            pt2_ids = pt2_ids[:max_pt2_len]

    return pt1_ids, pt2_ids


def assemble_masked_input(tokenizer, pt1_ids, pt2_ids, n_masked):
    """Lay out one masked-infilling input: [cls] pt1 [mask x n_masked] pt2 [end].

    Returns (input_ids, mask_start_idx). The wrap tokens are dropped for tokenizers that don't use
    them (the scratch model), which also shifts mask_start_idx by one.

    Shared deliberately: AntibodyInfillingDataset.__getitem__ builds TRAINING examples with this,
    and predict.py's build_masked_input builds INFERENCE inputs with it. The two must agree exactly
    — designing against a different layout than the model trained on is silently wrong, not a crash
    — and predict.py works from raw pt1/pt2 strings rather than dataset entries, so nothing else
    forces them to stay in step. One function is that guarantee.
    """
    mask_ids = [tokenizer.mask_id] * n_masked
    if tokenizer.uses_special_tokens:
        input_ids = [tokenizer.cls_id] + pt1_ids + mask_ids + pt2_ids + [tokenizer.end_id]
        return input_ids, 1 + len(pt1_ids)   # 1 for the leading cls token
    return pt1_ids + mask_ids + pt2_ids, len(pt1_ids)


class AntibodyInfillingDataset(Dataset):
    """One processed entry -> a masked-infilling example for the chosen CDR loop.

    Masks `target_region` on the chain that carries it and keeps the rest of that chain as context,
    optionally adding the paired partner chain and/or the antigen. Context assembly, length capping
    and label construction all happen in __getitem__.

    Args:
      json_path             -- processed JSON from preprocessing scripts
      tokenizer             -- TokenizerAdapter (NOT bare AntibodyTokenizer / HF tokenizer)
      max_len               -- hard cap on the assembled sequence; context is truncated to fit
      target_region         -- loop to mask: h1/h2/h3 (heavy) or l1/l2/l3 (light)
                                (also supported: cdr1/2/3, lcdr1/2/3 (old runs))
      include_partner_chain -- prepend the paired chain (light for a heavy target, heavy for light)
      include_antigen       -- add antigen context if available
    """

    def __init__(self, json_path, tokenizer, max_len=256, target_region="h3",
                 include_partner_chain=False, include_antigen=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_region = target_region
        self.include_partner_chain = include_partner_chain
        self.include_antigen = include_antigen

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _resolve_target(target_region):
        """Map a target_region string to (chain, cdr-key), e.g. "h3" -> ("heavy", "cdr3")"""
        tr = target_region.lower()
        chain = "light" if tr.startswith("l") else "heavy"   # covers l1/l2/l3 and lcdr1/2/3
        return chain, "cdr" + tr[-1]                          # last char is the CDR number

    @staticmethod
    def _split_regions(chain, region):
        """Split a chain dict into (pt1, target, pt2) around the requested cdr region"""
        if region == "cdr1":
            return chain['fwr1'], chain['cdr1'], chain['fwr2'] + chain['cdr2'] + chain['fwr3'] + chain['cdr3'] + chain['fwr4']
        elif region == "cdr2":
            return chain['fwr1'] + chain['cdr1'] + chain['fwr2'], chain['cdr2'], chain['fwr3'] + chain['cdr3'] + chain['fwr4']
        else:  # cdr3
            return chain['fwr1'] + chain['cdr1'] + chain['fwr2'] + chain['cdr2'] + chain['fwr3'], chain['cdr3'], chain['fwr4']

    def __getitem__(self, idx):
        """Assemble one example: mask the target loop, inject context, cap length, build labels.

        Returns {input_ids, labels, target_len}. `labels` carries the native residues at the masked
        positions and -100 everywhere else, so the loss is scored on the loop alone; `target_len`
        is the loop length, which the length-bucketed eval reports group by.
        """
        item = self.data[idx]

        # 1. Resolve which chain carries the target and which is the (optional) partner context
        target_chain_key, region = self._resolve_target(self.target_region)
        partner_key = "heavy" if target_chain_key == "light" else "light"

        pt1_seq, target_seq, pt2_seq = self._split_regions(item[target_chain_key], region)

        # 2. Tokenize target-chain parts (adapter.encode never adds special tokens)
        t = self.tokenizer
        pt1_ids = t.encode(pt1_seq)
        pt2_ids = t.encode(pt2_seq)
        target_ids = t.encode(target_seq)

        # --- CONTEXT INJECTION + LENGTH CAPPING --- (shared with predict.py; see inject_context)
        partner = item.get(partner_key) or {}
        antigen = item.get('antigen') or {}
        pt1_ids, pt2_ids = inject_context(
            t, pt1_ids, pt2_ids, len(target_ids),
            partner_seq=partner.get('full_aa') if self.include_partner_chain else None,
            antigen_seq=antigen.get('full_aa') if self.include_antigen else None,
            max_len=self.max_len,
        )

        # 3. Construct Final Input and Find Mask Start
        input_ids, mask_start_idx = assemble_masked_input(t, pt1_ids, pt2_ids, len(target_ids))

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

    [NOTE] pad_token_id defaults to 0, which is correct for AntibodyTokenizer's <pad> id, but
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
    dataset = AntibodyInfillingDataset("dataset/data/processed/processed_oas_data.json", adapter)
    collate = functools.partial(collate_fn, pad_token_id=adapter.pad_id)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate)

    # Peek at one batch
    for batch in dataloader:
        print("Input Shape:", batch["input_ids"].shape)
        print("First Input IDs:", batch["input_ids"][0])
        print("First Labels:", batch["labels"][0])
        break