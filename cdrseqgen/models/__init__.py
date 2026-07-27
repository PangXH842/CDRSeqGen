"""Models: the backbone registry (resolve) and the in-house architectures.

  * registry.py -- maps a model_type string to a Backbone (tokenizer + loaders + encoding
                   conventions). resolve() is the only entry point the rest of the repo uses.
  * custom/     -- architectures this repo implements, as opposed to checkpoints it downloads.

Adding a PLM is one register(Backbone(...)) call in registry.py; nothing else branches on model
family. Phase-2 structure models (IPA / equivariant GNN / diffusion) will land in custom/.
"""
