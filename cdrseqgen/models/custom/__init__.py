"""In-house model definitions — the architectures this repo owns, as opposed to loaded PLMs.

  * transformer.py -- the from-scratch encoder (CustomAntibodyConfig, CustomAntibodyModel,
                      build_model), reached as model_type "scratch" in the configs

Everything here is registered through ../registry.py, which is the only module that should import it
directly; train.py / predict.py / gen_eval.py all go through resolve() instead. Pretrained backbones
(ESM-2, IgBert, ProtBert) are loaded from Hugging Face by that registry and have no definition here.
"""
