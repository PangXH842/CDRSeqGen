"""Encoder-decoder view of the antibody data: two-stream span-infilling examples.

The third view, beside :mod:`cdrseqgen.data.dataset_mlm` (masked) and
:mod:`cdrseqgen.data.dataset_ar` (causal). All three build on the method-neutral
:mod:`cdrseqgen.data.dataset_common`, which owns framework/target splitting, context injection,
length capping and context dropout.

  * Seq2SeqTokenizer          -- the shared 25-token vocab plus <bos>/<eos>
  * s2s_adapter               -- TokenizerAdapter exposing bos_id/eos_id
  * build_encoder_input       -- framework parts -> the sequence the ENCODER reads
  * build_decoder_pair        -- target -> (decoder_input_ids, labels), teacher-forced
  * Seq2SeqInfillingDataset   -- one processed entry -> a two-stream example
  * s2s_collate_fn            -- pad both streams independently

Why a third method: the matched masked-vs-causal comparison (2026-09-04) ties on everything
native-referenced and separates only on judged plausibility, where causal leads 3.15 to 7.55. The
most economical explanation is the loop factorisation rather than the framework conditioning — a
masked LM's independent decoder samples every loop position from one pass, each ignorant of the
others, while a causal model conditions each residue on its predecessors. Encoder-decoder keeps
that factorisation and restores the one thing the masked side had: a bidirectional read of the
framework. It is the missing cell of the 2x2, not a new idea.

The layout is:

    encoder:  pt1  <mask>  pt2                     (read once, bidirectionally)
    decoder:  <bos>  target                        (input, teacher-forced)
    labels:          target  <eos>                 (shifted one position left)

**Two streams, not one sequence, and that is the whole structural difference from `ar`.** The causal
method writes ``pt1 <mask> pt2 <sep> target <eos>`` into a single sequence under one causal mask, so
(a) ``pt1`` is encoded without ever attending to ``pt2``, and (b) the answer must be budgeted for
inside ``max_len`` before it exists. Splitting the streams removes both:

  * **The framework is read bidirectionally.** FR1 and FR4 inform each other before a single residue
    is emitted, instead of only FR4 being informed by FR1. Note this is a difference in how the
    framework is *encoded*, not in what the decoder can see: the causal method already puts both
    flanks in the prompt, so `ar` conditions on FR4 too. Anyone reading this expecting "ar cannot
    see the right flank" has the wrong picture.

  * **``LOOP_LENGTH_RESERVE`` does not exist here, and that is a real fix rather than a tidy-up.**
    The causal prompt must hold back 96 positions of ``max_len`` for a loop it has not generated,
    and that reserve competes with optional context for the same budget: raising it 40 -> 96 cut the
    light chain the ``_lc`` runs exist to supply from ~96% retained to ~45%, silently, and forced
    the AR configs to ``max_len 384``. Here the encoder spends exactly ONE token on the gap marker
    and the decoder has an independent budget, so context gets the whole cap. An `s2s` config at
    ``max_len 256`` therefore holds strictly more context than an `ar` config at 256, which is worth
    remembering before putting the two in one table.

Length comes free, exactly as in `ar`: the decoder stops at ``<eos>``. Findings §10's gap-sentinel
classifier has no counterpart, and §9 already put the achievable ceiling at 11.07% / 3.01 MAE via a
non-neural V/J lookup, so length here is reported, not optimised.

Loss covers the decoder stream only. There is no ``-100`` prefix to mask out — unlike the causal
layout, the framework is not in the supervised stream at all, so "answer-only loss" is structural
rather than something the labels have to encode.

On the vocabulary: ``<bos>``/``<eos>`` are APPENDED at ids 25/26 and the first 25 ids are untouched,
matching :class:`cdrseqgen.data.dataset_ar.CausalTokenizer` exactly. The construction is duplicated
rather than imported so the two methods stay independently testable; ``test_dataset_s2s`` asserts
the two vocabularies agree token-for-token, which is the failure that duplication could cause. If a
fourth generative method needs it, move it to ``dataset_common`` then.
"""
import torch

from cdrseqgen.data.dataset_common import (AntibodyFrameworkDataset, AntibodyTokenizer,
                                           TokenizerAdapter, inject_context)


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


# The encoder spends exactly one non-residue position: the gap marker. Compare the causal layout's
# three (<mask>, <sep>, <eos>) — the separator is unnecessary when the answer is not in this stream,
# and the terminator belongs to the decoder.
ENCODER_LAYOUT_TOKENS = 1

# Longest loop the decoder may produce, and the cap on generated residues.
#
# Sized from data with headroom, NOT fitted to the corpus of the day — the causal side's equivalent
# constant was twice set to the exact maximum of the corpus in hand and twice broken by the next one
# (40 was the exact OAS max; SAbDab2 holds a 63-residue H3; OAS4 then arrived with a 49). Longest
# loop across every processed corpus on record is 63, so 96 is ~1.5x the worst case.
#
# Unlike LOOP_LENGTH_RESERVE this competes with NOTHING. It bounds the decoder stream only, so
# raising it costs decode steps and never trims a residue of framework or partner-chain context.
MAX_LOOP_RESIDUES = 96
# Decoding needs one further step for the terminator: a 40-residue loop is complete only once the
# 41st prediction emits <eos>.
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
    if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= S2S_DECODE_STEPS):
        raise ValueError(
            f"max_new_tokens must be an integer in [1, {S2S_DECODE_STEPS}]: the decoder allows "
            f"{MAX_LOOP_RESIDUES} residues plus one <eos> step")
    return max_new_tokens


def build_encoder_input(tokenizer, framework):
    """Return (encoder_ids, meta) for one framework: ``pt1 <mask> pt2``.

    The gap marker carries no length information — it is one token whether the loop is 5 residues or
    30 — which is what keeps length a decoder decision. Training and generation both call this, and
    it takes no generation-time parameter: the 2026-09-02 prompt-divergence bug came from exactly
    this construction site accepting a keyword that differed between the two, and the fix was to
    make the input a deterministic function of (framework, max_len) alone.
    """
    t = tokenizer
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

    if framework["max_len"] < ENCODER_LAYOUT_TOKENS:
        raise ValueError(
            f"max_len={framework['max_len']} cannot hold even the gap marker; raise it to >= 1")
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


def s2s_collate_fn(batch, pad_token_id=0):
    """Pad the encoder and decoder streams independently to their own longest example.

    The two streams have unrelated lengths — a batch can hold a 200-token framework with a 9-residue
    loop — so padding them to a shared maximum would waste most of the decoder. Both pad on the
    RIGHT, which is correct for teacher forcing because every supervised position carries a label
    and the pad positions are -100. Batched *generation* wants left-padded prompts, so a sampler
    must pad its own encoder inputs rather than reuse this.
    """
    def pad(key, value):
        return torch.nn.utils.rnn.pad_sequence(
            [item[key] for item in batch], batch_first=True, padding_value=value)

    input_ids = pad("input_ids", pad_token_id)
    decoder_input_ids = pad("decoder_input_ids", pad_token_id)

    # Explicit, not `input_ids != pad_token_id`: a real pad id may also be a real residue id.
    attention_mask = torch.zeros_like(input_ids)
    for row, item in enumerate(batch):
        attention_mask[row, :len(item["input_ids"])] = 1
    decoder_attention_mask = torch.zeros_like(decoder_input_ids)
    for row, item in enumerate(batch):
        decoder_attention_mask[row, :len(item["decoder_input_ids"])] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": pad("labels", -100),
    }
