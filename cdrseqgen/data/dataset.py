"""Dataset class script, preprocessed jsons -> torch tensors for training (MLM infilling).

Functions:
  * AntibodyTokenizer        -- custom 25-token vocab (20 amino acids + 5 special)
  * TokenizerAdapter         -- uniform tokenizer interface (custom, HF)
  * assemble_masked_input    -- the [cls] pt1 [mask x L] pt2 [end] layout
  * AntibodyInfillingDataset -- one entry -> (input_ids, labels, target_len) with one CDR masked
  * collate_fn               -- pad a list of items into a batch (+ attention mask)

Input schema:
    heavy / light            -- {fwr1..fwr4, cdr1..cdr3, full_aa, imgt_numbering}
    antigen (SAbDab2 only)   -- {full_aa, chains, type}
    split (optional)         -- "train"/"test", honored by get_datasets() in train.py

Notes:
  * Input layout: [cls] pt1 [mask x L] pt2 [end]; the wrap tokens are dropped for the scratch
    tokenizer (uses_special_tokens).
  * Loss is scored only on hidden loop residues. Full-span mode hides the entire CDR; mixed mode
    reveals a training-only subset and labels those visible positions -100.
  * Training and fixed-native evaluation use the native loop length as the mask count; bounded and
    inferred generation build their own prompts from framework_parts()
  * Partner chain context is prepended to framework
  * Where antigen context is injected depends on the backbone's position_type, not its model family
"""
import copy
import hashlib
import json

import torch
from torch.utils.data import Dataset, DataLoader

MASKING_POLICIES = ("full_span", "mixed")


def normalize_masking_policy(policy):
    """Return the canonical training masking policy name or raise a useful error."""
    normalized = str(policy).strip().lower().replace("-", "_")
    if normalized == "fullspan":
        normalized = "full_span"
    if normalized not in MASKING_POLICIES:
        raise ValueError(
            f"masking_policy must be one of {MASKING_POLICIES}, got {policy!r}"
        )
    return normalized


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

    Placement rules:
      * partner chain   -> PREPENDED to pt1
      * antigen         -> PREPENDED to pt1 for relative/rotary models (ESM-2)
                        -> APPENDED to pt2 for absolute-position models (scratch, BERT-style)
      * capping trims side carrying the antigen from its far end (???)
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
    budget = max(0, max_len - special_budget - n_target)

    if t.position_type == "relative":
        pt2_ids = pt2_ids[:budget]
        pt1_ids = pt1_ids[max(0, len(pt1_ids) - (budget - len(pt2_ids))):]
    else:  # absolute
        pt1_ids = pt1_ids[max(0, len(pt1_ids) - budget):]
        pt2_ids = pt2_ids[:max(0, budget - len(pt1_ids))]

    return pt1_ids, pt2_ids


def assemble_masked_input(tokenizer, pt1_ids, pt2_ids, n_masked):
    """Lay out one masked-infilling input: [cls] pt1 [mask x n_masked] pt2 [end].

    Returns (input_ids, mask_start_idx). The wrap tokens are dropped for tokenizers that don't use
    them (the scratch model), which also shifts mask_start_idx by one.
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
      masking_policy        -- full_span (default) or mixed training corruption
      mixed_full_span_probability -- fraction of mixed examples that still hide the entire loop
    """

    def __init__(self, json_path, tokenizer, max_len=256, target_region="h3",
                 include_partner_chain=False, include_antigen=False,
                 context_dropout=0.0, context_dropout_seed=0,
                 masking_policy="full_span", mixed_full_span_probability=0.5):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_region = target_region
        self.include_partner_chain = include_partner_chain
        self.include_antigen = include_antigen
        self.context_dropout = context_dropout
        self.context_dropout_seed = context_dropout_seed
        self.masking_policy = normalize_masking_policy(masking_policy)
        if not 0.0 <= mixed_full_span_probability <= 1.0:
            raise ValueError("mixed_full_span_probability must be between 0 and 1")
        self.mixed_full_span_probability = mixed_full_span_probability

    def masking_view(self, masking_policy, mixed_full_span_probability=None):
        """Return a lightweight dataset view sharing records but using another corruption policy."""
        view = copy.copy(self)
        view.masking_policy = normalize_masking_policy(masking_policy)
        if mixed_full_span_probability is not None:
            if not 0.0 <= mixed_full_span_probability <= 1.0:
                raise ValueError("mixed_full_span_probability must be between 0 and 1")
            view.mixed_full_span_probability = mixed_full_span_probability
        return view



    def _masked_target_offsets(self, target_len):
        """Choose target offsets hidden and supervised in this training example."""
        if (
            self.masking_policy == "full_span"
            or target_len <= 1
            or torch.rand(()).item() < self.mixed_full_span_probability
        ):
            return list(range(target_len))

        # A non-full mixed example reveals at least one and masks at least one residue. Sampling
        # the count uniformly exposes the iterative decoder's full range of partially filled states.
        masked_count = int(torch.randint(1, target_len, (1,)).item())
        return sorted(torch.randperm(target_len)[:masked_count].tolist())

    def _keep_context(self, idx, kind):
        """Whether THIS example keeps its `kind` ("partner" / "antigen") context, under context_dropout.

        Deterministic in idx, so an example's context regime is stable across epochs and runs.

        The point is to train ONE checkpoint that survives both regimes. Without it a model is
        locked to the regime it was trained in, in both directions: measured on H1/scratch, the
        with-context model drops 0.984 -> 0.261 when context is withheld, and the without-context
        model drops 0.997 -> 0.159 when context is supplied — for an absolute-position backbone a
        prepended partner chain shifts every learned position.

        Each context kind is drawn INDEPENDENTLY (the kind salts the hash), so a SAbDab2 run sees all
        four combinations — none / partner / antigen / both — rather than only "everything or
        nothing". That is what lets one checkpoint stand in for both the `_ag` and `_ag_lconly`
        configs instead of just being robust to total context loss.
        """
        if self.context_dropout <= 0.0:
            return True
        key = f"{self.context_dropout_seed}:{kind}:{idx}".encode()
        h = hashlib.blake2b(key, digest_size=8).digest()
        return (int.from_bytes(h, "big") / 2 ** 64) >= self.context_dropout

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

    def framework_parts(self, idx):
        """Return raw framework/target/context strings for inference-side prompt builders."""
        item = self.data[idx]
        target_chain_key, region = self._resolve_target(self.target_region)
        partner_key = "heavy" if target_chain_key == "light" else "light"
        pt1, target, pt2 = self._split_regions(item[target_chain_key], region)
        partner = item.get(partner_key) or {}
        antigen = item.get("antigen") or {}
        return {
            "pt1": pt1,
            "target": target,
            "target_length": len(target),
            "pt2": pt2,
            "partner_chain": partner.get("full_aa")
            if self.include_partner_chain and self._keep_context(idx, "partner") else None,
            "antigen": antigen.get("full_aa")
            if self.include_antigen and self._keep_context(idx, "antigen") else None,
            "max_len": self.max_len,
        }

    def __getitem__(self, idx):
        """Assemble one example: mask the target loop, inject context, cap length, build labels.

        Returns {input_ids, labels, target_len, masked_count}. `labels` carries native residues only
        at positions hidden in this view and -100 everywhere else. `target_len` remains the true loop
        length; evaluation uses a full-span view, so its mask count still equals that length.
        """
        framework = self.framework_parts(idx)
        pt1_seq, target_seq, pt2_seq = framework["pt1"], framework["target"], framework["pt2"]

        # 2. Tokenize target-chain parts (adapter.encode never adds special tokens)
        t = self.tokenizer
        pt1_ids = t.encode(pt1_seq)
        pt2_ids = t.encode(pt2_seq)
        target_ids = t.encode(target_seq)

        # --- CONTEXT INJECTION + LENGTH CAPPING --- (shared with generate.py; see inject_context)
        pt1_ids, pt2_ids = inject_context(
            t, pt1_ids, pt2_ids, len(target_ids),
            partner_seq=framework["partner_chain"], antigen_seq=framework["antigen"],
            max_len=self.max_len,
        )

        # 3. Construct Final Input and Find Mask Start
        input_ids, mask_start_idx = assemble_masked_input(t, pt1_ids, pt2_ids, len(target_ids))

        # 4. Optionally reveal native training residues. Evaluation always uses a separate
        # full_span view (pipeline.py), so no native target residue leaks into reported recovery.
        masked_offsets = set(self._masked_target_offsets(len(target_ids)))

        # 5. Create labels only for residues the model must reconstruct.
        labels = [-100] * len(input_ids)
        for i, t_id in enumerate(target_ids):
            position = mask_start_idx + i
            if i in masked_offsets:
                labels[position] = t_id
            else:
                input_ids[position] = t_id

        assert len(input_ids) == len(labels), f"Length Mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target_ids),
            "masked_count": len(masked_offsets),
        }


def framework_for_dataset(dataset, index):
    """Resolve a possibly nested ``Subset`` index to its dataset's raw framework strings."""
    while isinstance(dataset, torch.utils.data.Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    if not hasattr(dataset, "framework_parts"):
        raise TypeError("variable-length generation requires a dataset with framework_parts()")
    return dataset.framework_parts(index)


def collate_fn(batch, pad_token_id=0):
    """
    Takes a list of items and pads them to the same length so they fit in a Tensor.

    [NOTE] pad_token_id defaults to 0, which is correct for AntibodyTokenizer's <pad> id, but
    NOT for ESM-2's tokenizer (whose <pad> id is 1, while 0 is <cls>) — callers using an
    ESM-2 tokenizer must pass its real pad_token_id explicitly (see train.py/generate.py).
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
