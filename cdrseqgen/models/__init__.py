"""Models: the backbone registry (resolve) and the from-scratch transformer.

Adding a PLM is still one register(Backbone(...)) call in backbones.py; nothing else branches on
model family. Phase-2 structure models (IPA / equivariant GNN / diffusion) will land here too.
"""
