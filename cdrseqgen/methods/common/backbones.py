"""Shared checkpoint IO and protein tokenizer conventions; models stay method-owned."""
import os

from cdrseqgen.data.dataset_common import TokenizerAdapter


def load_custom_pretrained(path, model_cls, config_cls):
    from safetensors.torch import load_file
    model = model_cls(config_cls.from_pretrained(path))
    model.load_state_dict(load_file(os.path.join(path, 'model.safetensors')), strict=True)
    model.eval()
    return model


def esm_tokenizer(mt, pos, wrap):
    from transformers import EsmTokenizer
    return TokenizerAdapter.from_hf(
        EsmTokenizer.from_pretrained(mt), position_type=pos, uses_special_tokens=wrap)


def bert_tokenizer(mt, pos, wrap):
    from transformers import AutoTokenizer, BertTokenizer
    try:
        hf_tok = AutoTokenizer.from_pretrained(mt, use_fast=False)
    except Exception:
        hf_tok = BertTokenizer.from_pretrained(mt)
    return TokenizerAdapter.from_hf(
        hf_tok, position_type=pos, uses_special_tokens=wrap, space_residues=True)


def tokenizer_source(model_type, checkpoint_path=None):
    """Prefer the exact saved vocabulary; retain support for older runs without tokenizer files."""
    if checkpoint_path and os.path.isfile(os.path.join(checkpoint_path, "tokenizer_config.json")):
        return str(checkpoint_path)
    return model_type
