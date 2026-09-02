"""Causal view of the antibody data: span-infilling examples for an autoregressive method.

The counterpart of :mod:`cdrseqgen.data.dataset_mlm` (the masked-LM view). Both build on the
method-neutral :mod:`cdrseqgen.data.dataset_common`, which owns framework/target splitting, context
injection, length capping and context dropout.

  * CausalTokenizer          -- the shared 25-token vocab plus <bos>/<eos>
  * causal_adapter           -- TokenizerAdapter exposing eos_id
  * build_causal_prompt      -- framework parts -> the prompt a causal model conditions on
  * CausalInfillingDataset   -- one processed entry -> (input_ids, labels) for causal training
  * causal_collate_fn        -- pad a batch, right-aligned, with an answer mask

Why an autoregressive method at all: three Phase-1 findings converge on the masked-LM *objective
class* rather than anything tunable inside it — iterative and independent decoding are within noise
because the learned marginals are near-independent (Step 4), the model's own confidence barely
tracks its correctness (§12), and re-weighting the loss across loop positions moves nothing (§13).
The one axis where IgLM — autoregressive — beats CDRSeqGen decisively is judged plausibility (§11).

The layout is:

    pt1  <mask>  pt2  <sep>  target  <eos>
    └─────── prompt ───────┘ └──── answer ────┘

**The reordering is the whole point, and the naive layout is wrong.** Writing the sequence in its
natural order — ``pt1 target pt2`` — means the model has not read ``pt2`` at the moment it generates
the loop, so it cannot condition on the right flank at all. That flank is not decoration: FR4 carries
the J-gene motif that terminates CDR-H3, and Findings §11 measured what happens without it — when
IgLM was forced past its natural stopping point only 2% of loops ended in the J-gene motif against
20% when it chose freely. Putting the corrupted sequence first and the answer last is how IgLM does
it (`benchmark/adapters/iglm.py` prompts with ``mask_span(parent_sequence, start, end)``), and it is
the standard span-infilling trick.

Two consequences worth stating:

  * **Length comes free.** The answer ends at ``<eos>``, so the model chooses its own loop length.
    The gap-sentinel classifier of Findings §10 — which sits *below* the "always predict 15" baseline
    — exists only because a masked LM cannot do this. There is nothing to port.
  * **Loss covers the answer only.** The prompt is given, so supervising it would train the model to
    reproduce framework residues it was handed, inflating any token-level number. ``labels`` is -100
    across the prompt and carries the loop plus its ``<eos>``.

On the vocabulary: ``<bos>``/``<eos>`` are APPENDED at ids 25/26 and the first 25 ids are untouched.
Every existing MLM checkpoint has a 25-row embedding table keyed to those ids; renumbering would
silently mismatch all of them. The shared vocab stays a prefix of the causal one.
"""
import torch

from cdrseqgen.data.dataset_common import (AntibodyFrameworkDataset, AntibodyTokenizer,
                                       TokenizerAdapter, inject_context)


BOS = "<bos>"
EOS = "<eos>"


class CausalTokenizer(AntibodyTokenizer):
    """The shared vocab, extended with <bos>/<eos> at the end so ids 0-24 keep their meaning."""

    def __init__(self):
        super().__init__()
        for token in (BOS, EOS):
            if token in self.tok_to_id:                       # defensive: never renumber
                raise ValueError(f"{token} already in the shared vocab; ids would shift")
            self.tok_to_id[token] = len(self.vocab)
            self.id_to_tok[len(self.vocab)] = token
            self.vocab.append(token)
        self.bos_id = self.tok_to_id[BOS]
        self.eos_id = self.tok_to_id[EOS]


def causal_adapter(tokenizer=None, position_type="absolute"):
    """A TokenizerAdapter for causal use: no mask_id required, <eos> as the answer terminator.

    `mask_id` is still supplied — the span-infilling layout marks the gap with <mask> in the prompt,
    which is what lets the model see BOTH flanks before it generates. It is the *adapter's*
    requirement that was relaxed when `dataset_common.py` was split out, not the layout's use of a
    sentinel.
    """
    tok = tokenizer or CausalTokenizer()
    adapter = TokenizerAdapter(
        encode=tok.encode,
        decode=tok.decode,
        vocab_size=len(tok.vocab),
        pad_id=tok.pad_id,
        mask_id=tok.mask_id,
        cls_id=None,
        end_id=None,
        chain_sep_id=tok.tok_to_id["<sep>"],
        special_ids={tok.tok_to_id[t]
                     for t in ("<pad>", "<unk>", "<cls>", "<sep>", "<mask>", BOS, EOS)},
        uses_special_tokens=False,
        position_type=position_type,
    )
    adapter.bos_id = tok.bos_id
    adapter.eos_id = tok.eos_id
    return adapter


# prompt <mask> + prompt <sep> + answer <eos>
LAYOUT_SPECIAL_TOKENS = 3

# How much room the prompt leaves for the answer when capping context to max_len.
#
# FIXED, and used identically by training and generation. It was previously the native loop length
# at training and `max_new_tokens` at generation, which made the prompt a function of a
# generation-time parameter: with max_len=128 the same framework produced an 87-token prompt at
# sampling against 110 at training, so the model met a framework 23 tokens shorter than anything it
# had trained on and never reached a state where <eos> was likely (P(eos) ~ 1e-4 free-running
# against 0.91 teacher-forced). A constant makes the prompt a deterministic function of
# (framework, max_len) alone, which is the only way generation can reproduce the training prompt
# without knowing the length it is trying to predict.
#
# 40 covers every CDR loop in both datasets (H3 tops out around 34).
LOOP_LENGTH_RESERVE = 40


def build_causal_prompt(tokenizer, framework, *, reserve=LOOP_LENGTH_RESERVE):
    """Return (prompt_ids, meta) for one framework: ``pt1 <mask> pt2 <sep>``.

    `reserve` is how many answer positions the cap keeps room for. **Leave it at the default.** It
    is deliberately a constant rather than the native length, so that the prompt a model trains on
    and the prompt it generates from are byte-identical — see LOOP_LENGTH_RESERVE. Passing the
    native length here would also leak it into how hard the flanks are trimmed, which is exactly the
    information a length-free track must not receive.
    """
    t = tokenizer
    pt1_ids = t.encode(framework["pt1"])
    pt2_ids = t.encode(framework["pt2"])

    # Reserve the answer AND the three layout tokens, so a capped prompt still leaves room to finish.
    pt1_ids, pt2_ids = inject_context(
        t, pt1_ids, pt2_ids, reserve + LAYOUT_SPECIAL_TOKENS,
        partner_seq=framework["partner_chain"], antigen_seq=framework["antigen"],
        max_len=framework["max_len"],
    )
    prompt = pt1_ids + [t.mask_id] + pt2_ids + [t.chain_sep_id]

    # The flanks are truncatable; the answer is not. Once context is capped to nothing the prompt is
    # just <mask><sep>, and a max_len below reserve+3 can only be honoured by cutting the loop the
    # model is supposed to learn. Silently returning an over-length example would surface much later
    # as a position-embedding overflow, so fail here, where the cause is named.
    minimum = reserve + LAYOUT_SPECIAL_TOKENS
    if framework["max_len"] < minimum:
        raise ValueError(
            f"max_len={framework['max_len']} cannot hold a {reserve}-residue loop: the answer plus "
            f"its <mask>/<sep>/<eos> needs at least {minimum} positions, and truncating it would "
            f"corrupt the training target. Raise max_len to >= {minimum}."
        )
    if framework["target_length"] > reserve:
        raise ValueError(
            f"loop of {framework['target_length']} residues exceeds the {reserve}-position reserve, "
            f"so the assembled example could overflow max_len. Raise LOOP_LENGTH_RESERVE.")
    return prompt, {"pt1_len": len(pt1_ids), "pt2_len": len(pt2_ids), "reserve": reserve}


class CausalInfillingDataset(AntibodyFrameworkDataset):
    """One processed entry -> a causal span-infilling example, supervised on the answer only."""

    def __getitem__(self, idx):
        framework = self.framework_parts(idx)
        t = self.tokenizer
        prompt, meta = build_causal_prompt(t, framework)
        target_ids = t.encode(framework["target"])
        answer = target_ids + [t.eos_id]

        input_ids = prompt + answer
        labels = [-100] * len(prompt) + answer      # the prompt is given, not predicted

        assert len(input_ids) == len(labels)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt),
            "target_len": len(target_ids),
            "pt1_len": meta["pt1_len"],
        }


def causal_collate_fn(batch, pad_token_id=0):
    """Pad a batch to its longest example.

    Padding is on the RIGHT, which is correct for teacher-forced training because every supervised
    position is already labelled and the pad positions are -100. Batched *generation* wants
    left-padding instead, so a sampler should pad its own prompts rather than reuse this.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100)
    attention_mask = torch.zeros_like(input_ids_padded)
    for row, item in enumerate(input_ids):
        attention_mask[row, :len(item)] = 1     # explicit: a real pad id may also be a real residue
    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "attention_mask": attention_mask,
        "prompt_lengths": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "target_lengths": torch.tensor([b["target_len"] for b in batch], dtype=torch.long),
    }
