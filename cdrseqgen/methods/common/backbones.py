"""Checkpoint IO shared by scratch backbones; architecture construction stays method-owned."""
import os


def load_custom_pretrained(path, model_cls, config_cls):
    from safetensors.torch import load_file
    model = model_cls(config_cls.from_pretrained(path))
    model.load_state_dict(load_file(os.path.join(path, 'model.safetensors')), strict=True)
    model.eval()
    return model
