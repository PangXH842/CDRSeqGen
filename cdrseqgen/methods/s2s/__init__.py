"""Encoder-decoder span infilling — the third method, beside ``methods/mlm`` and ``methods/ar``.

Why it exists: the matched masked-vs-causal comparison (2026-09-04) ties on every native-referenced
metric and separates only on judged plausibility, where causal leads 3.15 to 7.55. The economical
explanation is the loop factorisation rather than the framework conditioning — a masked LM samples
every loop position from one pass, each ignorant of the others, while a causal model conditions each
residue on its predecessors. This method keeps that factorisation and restores the bidirectional
framework read the masked side had, which makes it the missing cell of a 2x2:

    method    framework conditioning   loop generation              length
    mlm       bidirectional            parallel over a fixed span   committed up front
    ar        causal                   left-to-right                emergent (<eos>)
    s2s       bidirectional            left-to-right                emergent (<eos>)

  * models/transformer.py -- from-scratch encoder-decoder, model_type "scratch"
  * backbones.py          -- seq2seq backbone registry, resolve(model_type) -> Seq2SeqBackbone

The data view lives in :mod:`cdrseqgen.data.dataset_s2s`, beside the masked and causal ones, because
``cdrseqgen/data/`` holds all data code.

**On the expected result.** Findings §4 measured no benefit from partner-chain or antigen context,
and `mlm` and `ar` tie on recovery, so the prior against "better framework conditioning helps
recovery" is strong and this method may well close that axis negatively. The durable argument for
the family is the backbone: an encoder-decoder is the only one of the three where a natively
span-infilling PRETRAINED model (ProtT5) drops in without bolting randomly-initialised sentinels
onto something that never saw the format — the surgery ``methods/ar/backbones.py`` documents for
``hf_decoder`` — and without fine-tuning IgLM's weights, which are the §11 baseline.
"""
