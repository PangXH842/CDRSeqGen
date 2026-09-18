"""Encoder-decoder view of the antibody data: two-stream span-infilling examples.

Default layout:

    encoder:  pt1  <mask>  pt2                     (read once, bidirectionally)
    decoder:  <bos>  target                        (input, teacher-forced)
    labels:          target  <eos>                 (shifted one position left)

Seq2SeqInfillingDataset:
  * __getitem__     -- encoder stream + teacher-forced decoder pair, with the loop
                       checked against the decoder's residue budget
  * (remaining methods inherited from AntibodyFrameworkDataset)

Seq2SeqTokenizer:
  * the shared 25-token vocab with <bos>/<eos> appended at 25/26, so ids 0-24 keep
    their meaning and existing checkpoints' embedding tables still match

Utility functions:
  * s2s_adapter                 -- TokenizerAdapter exposing bos_id/eos_id
  * build_encoder_input         -- framework parts -> the sequence the ENCODER reads
  * build_decoder_pair          -- target -> (decoder_input_ids, labels), teacher-forced
  * s2s_collate_fn              -- pad both streams independently
  * validate_s2s_decode_steps   -- bound sampling by the decoder's residue budget

Notes:
  * No LOOP_LENGTH_RESERVE here: the encoder spends one token on the gap marker and the
    decoder has its own budget, so context gets the whole cap. An s2s config at max_len 256
    holds strictly more context than an ar config at 256 — worth remembering before putting
    the two in one table.
  * Loss covers the decoder stream only, structurally: the framework is never in the
    supervised stream, so there is no -100 prefix for the labels to encode.
  * The vocab construction is duplicated from dataset_ar rather than imported, to keep the
    two methods independently testable. `test_dataset_s2s` asserts they agree token-for-token
    — that is the failure duplication could cause. Move it to dataset_common if a fourth
    method needs it.
"""

import torch

from cdrseqgen.datasets.dataset_common import (AntibodyFrameworkDataset, AntibodyTokenizer,
                                           TokenizerAdapter, inject_context,
                                            pad_batch, length_mask, validate_decode_steps)


BOS = "<bos>"
EOS = "<eos>"


class Seq2SeqTokenizer(AntibodyTokenizer):
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


ENCODER_LAYOUT_TOKENS = 1
MAX_LOOP_RESIDUES = 96
S2S_DECODE_STEPS = MAX_LOOP_RESIDUES + 1

def s2s_adapter(tokenizer=None, position_type="relative"):
    """A TokenizerAdapter for encoder-decoder use, exposing <bos>/<eos>.

    Defaults to ``relative`` because the scratch backbone uses rotary positions: the encoder stream
    varies in length with optional context, and Findings §8 measured what absolute positions do to a
    checkpoint whose context regime shifts (0.984 -> 0.261 when context is withheld).
    """
    tok = tokenizer or Seq2SeqTokenizer()
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


def validate_s2s_decode_steps(max_new_tokens):
    """Return a decode cap that fits the decoder's residue budget."""
    return validate_decode_steps(max_new_tokens, S2S_DECODE_STEPS,
                                 f"the decoder allows {MAX_LOOP_RESIDUES} residues plus one <eos> step")


def build_encoder_input(tokenizer, framework):
    """Return (encoder_ids, meta) for one framework: ``pt1 <mask> pt2``.

    A PLM adapter carries encoder_tokenizer separately from its residue decoder tokenizer.
    Native CLS/end wrappers count against max_len, and optional context uses native encoder ids.

    The gap marker carries no length information — it is one token whether the loop is 5 residues or
    30 — which is what keeps length a decoder decision. Training and generation both call this, and
    it takes no generation-time parameter: the 2026-09-02 prompt-divergence bug came from exactly
    this construction site accepting a keyword that differed between the two, and the fix was to
    make the input a deterministic function of (framework, max_len) alone.
    """
    t = getattr(tokenizer, "encoder_tokenizer", tokenizer)
    pt1_ids = t.encode(framework["pt1"])
    pt2_ids = t.encode(framework["pt2"])

    # Only the gap marker is reserved. The loop itself never occupies an encoder position, so
    # context is capped against the full max_len rather than max_len minus a 96-residue answer.
    pt1_ids, pt2_ids = inject_context(
        t, pt1_ids, pt2_ids, ENCODER_LAYOUT_TOKENS,
        partner_seq=framework["partner_chain"], antigen_seq=framework["antigen"],
        max_len=framework["max_len"],
    )
    encoder_ids = pt1_ids + [t.mask_id] + pt2_ids

    if t.uses_special_tokens:
        encoder_ids = [t.cls_id] + encoder_ids + [t.end_id]

    minimum = ENCODER_LAYOUT_TOKENS + (2 if t.uses_special_tokens else 0)
    if framework["max_len"] < minimum:
        raise ValueError(
            f"max_len={framework['max_len']} cannot hold the gap marker and wrappers; "
            f"raise it to >= {minimum}")
    return encoder_ids, {"pt1_len": len(pt1_ids), "pt2_len": len(pt2_ids)}


def build_decoder_pair(tokenizer, target_ids):
    """Return (decoder_input_ids, labels) for a teacher-forced loop.

    ``<bos> t0 t1 ... tn`` in, ``t0 t1 ... tn <eos>`` out — equal length, offset by one, so position
    i predicts token i+1. The shift is done HERE rather than inside the model: a model that shifts
    its own labels and a caller that also shifts produce a silent off-by-one that still trains, just
    to the wrong target, and the resulting checkpoint looks merely bad rather than broken.
    """
    target_ids = list(target_ids)
    return [tokenizer.bos_id] + target_ids, target_ids + [tokenizer.eos_id]


class Seq2SeqInfillingDataset(AntibodyFrameworkDataset):
    """One processed entry -> a two-stream encoder-decoder example."""

    def __getitem__(self, idx):
        framework = self.framework_parts(idx)
        t = self.tokenizer
        encoder_ids, meta = build_encoder_input(t, framework)
        target_ids = t.encode(framework["target"])

        if len(target_ids) > MAX_LOOP_RESIDUES:
            raise ValueError(
                f"loop of {len(target_ids)} residues exceeds the {MAX_LOOP_RESIDUES}-residue "
                f"decoder budget, so it could never be generated. Raise MAX_LOOP_RESIDUES.")

        decoder_input_ids, labels = build_decoder_pair(t, target_ids)
        assert len(decoder_input_ids) == len(labels)
        return {
            "input_ids": torch.tensor(encoder_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_len": len(target_ids),
            "pt1_len": meta["pt1_len"],
        }


def s2s_collate_fn(batch, pad_token_id=0, encoder_pad_token_id=None):
    """Pad the encoder and decoder streams independently to their own longest example.

    The two streams have unrelated lengths — a batch can hold a 200-token framework with a 9-residue
    loop — so padding them to a shared maximum would waste most of the decoder. Both pad on the
    RIGHT, which is correct for teacher forcing because every supervised position carries a label
    and the pad positions are -100. PLM encoders can use a different pad id from the residue decoder.
    Sampling encodes one unpadded framework at a time and shares it across candidates.
    """
    encoder_pad = pad_token_id if encoder_pad_token_id is None else encoder_pad_token_id
    input_ids = pad_batch(batch, "input_ids", encoder_pad)
    decoder_input_ids = pad_batch(batch, "decoder_input_ids", pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": length_mask(input_ids, batch, "input_ids"),
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": length_mask(decoder_input_ids, batch, "decoder_input_ids"),
        "labels": pad_batch(batch, "labels", -100),
    }
