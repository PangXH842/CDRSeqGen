"""Method-neutral antibody data: tokenizers, framework/target splitting, and context injection.

Everything here is shared by every generation method, because none of it knows how a loop will be
produced. It answers only: which residues are the target, which are the surrounding framework, what
paired-chain or antigen context comes with them, and how that context is capped to `max_len`.

  * AntibodyTokenizer        -- custom 25-token vocab (20 amino acids + 5 special)
  * TokenizerAdapter         -- uniform tokenizer interface (custom, HF)
  * inject_context           -- add partner-chain / antigen context, then cap to max_len
  * AntibodyFrameworkDataset -- processed JSON -> (pt1, target, pt2, context) per entry
  * framework_for_dataset    -- one entry's framework parts, by index

Split out of ``dataset_mlm.py`` on 2026-08-31 so a non-masked method can reuse it. The masked-LM view --
masking policies, ``assemble_masked_input``, ``AntibodyInfillingDataset`` and its collator -- stays
in ``dataset_mlm.py`` and builds on ``AntibodyFrameworkDataset``. An autoregressive or encoder-decoder
method subclasses the same base and lays its own inputs out, rather than inheriting a mask token it
has no use for.

Note ``mask_id`` on the adapter is optional for exactly that reason: a causal tokenizer need not
have a mask token, and requiring one was the concrete thing coupling this layer to masked LMs.
"""
import copy
import hashlib
import json
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


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

    def __init__(self, *, encode, decode, vocab_size, pad_id, cls_id, end_id,
                 chain_sep_id, special_ids, uses_special_tokens, position_type, mask_id=None):
        self.encode = encode
        self.decode = decode
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        # None for a causal tokenizer with no mask token; masked methods require it.
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

        adapter = cls(
            encode=_encode,
            decode=lambda ids: tok.decode(ids).replace(" ", ""),
            # ``vocab_size`` excludes runtime-added special tokens for several HF tokenizers;
            # ``len(tok)`` is the embedding-table size callers actually need.
            vocab_size=len(tok),
            pad_id=tok.pad_token_id,
            mask_id=tok.mask_token_id,
            cls_id=cls_id,
            end_id=end_id,
            chain_sep_id=end_id,
            special_ids=set(tok.all_special_ids),
            uses_special_tokens=uses_special_tokens,
            position_type=position_type,
        )
        adapter.save_pretrained = tok.save_pretrained
        adapter.backend_tokenizer = tok
        return adapter


def inject_context(tokenizer, pt1_ids, pt2_ids, n_target, *,
                   partner_seq=None, antigen_seq=None, max_len=256):
    """Add partner-chain / antigen context around the target, then cap to max_len.

    Returns the (pt1_ids, pt2_ids) a method then lays out. `partner_seq`/`antigen_seq` are raw
    residue strings (None = that context is off). `n_target` is how many positions the target will
    occupy once laid out, so the cap leaves room for it whether those are mask tokens or generated
    residues.

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


class AntibodyFrameworkDataset(Dataset):
    """One processed entry -> the target loop and the framework/context around it.

    Method-neutral: it locates `target_region` on the chain that carries it, keeps the rest of that
    chain as context, and optionally supplies the paired partner chain and/or the antigen. It does
    NOT lay out model inputs — a method subclasses this and implements `__getitem__`.

    Args:
      json_path             -- processed JSON from preprocessing scripts
      tokenizer             -- TokenizerAdapter (NOT bare AntibodyTokenizer / HF tokenizer)
      max_len               -- hard cap on the assembled sequence; context is truncated to fit
      target_region         -- loop to target: h1/h2/h3 (heavy) or l1/l2/l3 (light)
                                (also supported: cdr1/2/3, lcdr1/2/3 (old runs))
      include_partner_chain -- supply the paired chain (light for a heavy target, heavy for light)
      include_antigen       -- supply antigen context if available
      context_dropout       -- fraction of examples that drop each context kind, drawn per kind
    """

    def __init__(self, json_path, tokenizer, max_len=256, target_region="h3",
                 include_partner_chain=False, include_antigen=False,
                 context_dropout=0.0, context_dropout_seed=0):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_region = target_region
        self.include_partner_chain = include_partner_chain
        self.include_antigen = include_antigen
        self.context_dropout = context_dropout
        self.context_dropout_seed = context_dropout_seed

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


def framework_for_dataset(dataset, index):
    """Resolve a possibly nested ``Subset`` index to its dataset's raw framework strings."""
    while isinstance(dataset, torch.utils.data.Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    if not hasattr(dataset, "framework_parts"):
        raise TypeError("variable-length generation requires a dataset with framework_parts()")
    return dataset.framework_parts(index)


def evaluation_context_view(dataset, *, include_partner_chain=None):
    """Share records/tokenizer while disabling stochastic training context dropout.

    Preserve Subset ordering, including nested subsets, so context comparisons use identical
    examples. Never mutate a training dataset or the view used by the other evaluation arm.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return torch.utils.data.Subset(
            evaluation_context_view(dataset.dataset, include_partner_chain=include_partner_chain),
            dataset.indices)
    if not isinstance(dataset, AntibodyFrameworkDataset):
        raise TypeError("Context evaluation requires an AntibodyFrameworkDataset")
    view = copy.copy(dataset)
    view.context_dropout = 0.0
    if include_partner_chain is not None:
        view.include_partner_chain = bool(include_partner_chain)
    return view


def select_split_indices(records, eval_split, *, data_path, seed, test_size=0.1):
    """(train_idx, eval_idx, eval_label) for a processed dataset's named splits.

    Shared by every method rather than reimplemented per method, and deliberately so: this is the
    guard added on 2026-08-18 after an unset ``eval_split`` had been selecting checkpoints on the
    **test** partition in every batch through 08-17, which is still the largest caveat in the
    README. A second copy that drifted would recontaminate one method's results silently, and the
    logic has no method-specific content — it reads only each record's ``split`` field.
    """
    selected = (eval_split or "").strip()
    if records and all("split" in entry for entry in records):
        available = sorted({entry["split"] for entry in records})
        train_idx = [i for i, entry in enumerate(records) if entry["split"] == "train"]
        if selected:
            if selected == "train" or selected not in available:
                raise ValueError(
                    f"eval_split={selected!r} is invalid; available non-training splits: "
                    f"{[s for s in available if s != 'train']}")
            eval_idx = [i for i, entry in enumerate(records) if entry["split"] == selected]
        elif "test" in available:
            # Early stopping and load_best_model_at_end both read this split, so an unset
            # eval_split used to select checkpoints on the final holdout without saying so.
            raise ValueError(
                f"eval_split is unset and {data_path} carries a 'test' split, so every non-train "
                "entry — the test partition — would drive early stopping and best-checkpoint "
                "selection. Declare it: 'validation' while tuning, 'test' only once the settings "
                f"are frozen. Available non-training splits: "
                f"{[s for s in available if s != 'train']}")
        else:
            # No test partition to protect: every non-train entry is the evaluation set.
            eval_idx = [i for i, entry in enumerate(records) if entry["split"] != "train"]
        return train_idx, eval_idx, (selected or "all non-train")

    if selected:
        raise ValueError("eval_split requires a processed dataset with named split fields")
    from sklearn.model_selection import train_test_split
    train_idx, eval_idx = train_test_split(range(len(records)), test_size=test_size,
                                           random_state=seed)
    return list(train_idx), list(eval_idx), "random split"
