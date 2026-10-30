<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES, and DO NOT COMMIT ON MY BEHALF -->
# Operation Reclamation

[Back to README](../README.md) · [Project roadmap](ROADMAP.md) ·
[Method workflow map](../cdrseqgen/methods/common/README.md) · [Current handoff](PROJECT_STATUS.md)

This is the manual code-review checklist, moved from the README on **26 October 2026**.
Review proceeds through MLM, AR and S2S before generalizing the shared workflow.
Checkmarks record your own understanding and review; passing tests or an AI refactor
does not complete an item. The protection boundaries below still apply.

<!-- MANUAL PROTECTION ZONE BEGINS -->
- **Operation Reclamation** -- Manual code cleaning: Comprehend and rewrite the project content in my own terms - previous work handled by AI resulted in the project almost becoming a "black box" for the developer himself 
  - [ ] `benchmark/`
    - [ ] `adapters/` 
    - [ ] `_common`, `bootstrap`, `export_manifest`, `scorer`
    - [ ] `plot_comparison`
    - [ ] README
  - [ ] `cdrseqgen/` 
    - [x] `datasets/`
      - [x] `preprocess/` (1006)
      - [x] `dataset_<method>` (1006)
    - [x] `eval/` (`generative`, `metrics`) (1006)
    - [x] `methods/mlm/` (1029) 
      - [x] custom model (`models/transformer`) (1022)
      - [x] training block (1027)
        - [x] `train_cli`, `training` (1027)
        - [x] `arguments` (1027)
        - [x] `backbones` (1022)
        - [x] `losses` (1022)
        - [x] `recovery` (1027) 
      - [x] evaluation/generation block (1029) 
        - [x] `eval_cli`, `evaluation` (1029) 
        - [x] `gen_cli`, `generation` (1029)  
        - [x] `sampling` (1029) 
        - [x] `length` (1029) 
    - [ ] `methods/ar/`
      - [ ] custom model (`models/transformer`)
      - [ ] `train_cli`, `training`
      - [ ] `arguments` 
      - [ ] `backbones`
      - [ ] `eval_cli`, `evaluation`
      - [ ] `gen_cli`, `generation`
      - [ ] `sampling`
    - [ ] `methods/s2s/`
      - [ ] custom model (`models/transformer`)
      - [ ] `train_cli`, `training`
      - [ ] `arguments` 
      - [ ] `backbones`
      - [ ] `eval_cli`, `evaluation`
      - [ ] `gen_cli`, `generation`
      - [ ] `sampling`
      - [ ] `diagnostics` (??)
    - [ ] `methods/common/`
      - [ ] `arguments` 
      - [ ] `backbones`
      - [ ] `callbacks`
      - [ ] `eos_decoding` (AR/S2S shared)
      - [ ] `evaluation`
      - [ ] `logging`, `run_setup`, `record_results` 
      - [ ] `losses`
      - [ ] `sampling`
      - [ ] `training`
    - [ ] `visualization/` 
  - [ ] `unit_tests/`
  - [ ] root scripts
  - [ ] README
  - [ ] local `results/`

<!-- MANUAL PROTECTION ZONE ENDS -->
