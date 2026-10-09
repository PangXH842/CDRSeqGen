# Experiment results by question

The category view at `results/by_experiment/` is generated from
[`results_catalog.json`](results_catalog.json) with
`/home/pang/miniconda3/envs/cdrprj/bin/python scripts/organize_results.py` (or any Python 3.11+).
It uses relative symlinks. The original timestamped folders in `results/runs/` remain the
source records: their saved `run_config.json`, `plan.json`, and internal paths are not rewritten.
A batch can appear in several categories. Hyperparameter sweeps are excluded from this view;
their summaries remain in `results/runs/*sweep*/`. This is an inventory, not a claim that
experiments within a category used identical protocols.

The October 7 control and MLM dropout category links are available. Full catalog regeneration
still needs the historical `results/diagnostics/20260914_igbert` and
`results/diagnostics/20260915_igbert_summary` folders restored; their catalog
entries are preserved.

| Experiment type | Results to read | What the evidence covers |
| --- | --- | --- |
| Scratch baseline | [validation OAS matrix](results/by_experiment/scratch_baseline/validation_oas_matrix/batch_tables.md), [AR length-corrected](results/by_experiment/scratch_baseline/ar_length_corrected/batch_tables.md), [S2S pair](results/by_experiment/scratch_baseline/s2s_scratch_pair/batch_tables.md), [three-method comparison](results/by_experiment/scratch_baseline/three_method_comparison/batch_tables.md) | Completed exploratory MLM/AR/S2S runs. The earliest 07-24 and 07-29 matrices and pre-validation 08-17 matrix are also indexed, but selected on the test partition or predate the validation-safe OAS rerun. |
| Pretrained models | [S2S encoder matrix](results/by_experiment/pretrained_models/s2s_encoder_matrix/batch_tables.md), [validation OAS matrix](results/by_experiment/pretrained_models/validation_oas_matrix/batch_tables.md) | ESM-2/IgBert/ProtBert S2S encoders and older MLM ESM-2; no pretrained AR arm. These are system comparisons, not an isolated pretraining effect. |
| Partner context | [three-method comparison](results/by_experiment/partner_context/three_method_comparison/batch_tables.md), [S2S encoder matrix](results/by_experiment/partner_context/s2s_encoder_matrix/batch_tables.md), [six AR/S2S no-partner controls](results/reviews/20261007-no-partner-controls.md), [two MLM controls](results/reviews/20261007-mlm-no-partner-controls.md), [context review](results/reviews/0915-0917-review.md) | Eight no-partner training controls are audited: scratch MLM, scratch AR, scratch S2S and ESM-2 S2S at seeds 42/43. MLM returns 6,000 candidates without an attempt-validity ledger. The new MLM dropout comparison completes six matched checkpoints and ten evaluations; [audit](results/reviews/20261007-mlm-dropout-screen.md). The September AR/S2S screen uses earlier source/runtime versions. |
| Context dropout | [AR/S2S screen](results/by_experiment/context_dropout/ar_s2s_screen/batch_tables.md), [October confirmation audit](results/reviews/20261007-esm2-confirmation.md) | AR/S2S screen at seeds 42/43, four audited ESM-2 trials at seeds 44/45, and the [completed four-run MLM screen](results/reviews/20261007-mlm-dropout-screen.md), each evaluated in both regimes. Historical and new batches retain their distinct source/runtime provenance. |
| Pretrained vs random initialization | — | No same-architecture control yet. |
| Frozen encoder | [IgBert diagnostic](results/by_experiment/frozen_encoder/igbert_three_arm_diagnostic/batch_tables.md) | Three-epoch frozen/trainable diagnostic at one seed; generation from the saved checkpoints is pending. |
| Conditioning/collapse diagnostics | [collapse report](results/by_experiment/conditioning_diagnostics/igbert_collapse_report/REPORT.md), [diagnostic summary](results/by_experiment/conditioning_diagnostics/igbert_diagnostic_summary/REPORT.md) | One collapse and one same-seed non-reproduction; incidence and a preventive intervention are unknown. |
| Length-free generation | [three-method comparison](results/by_experiment/length_free_generation/three_method_comparison/batch_tables.md), [S2S encoder matrix](results/by_experiment/length_free_generation/s2s_encoder_matrix/batch_tables.md) | EOS/inferred-length results and length analysis. The assembled three-method comparison is explicitly marked as spliced in its `PROVENANCE.md`; use the fresh comparison for headline analysis. |
| Decoding/sampling | [validation OAS matrix](results/by_experiment/decoding_sampling/validation_oas_matrix/batch_tables.md) | Historical MLM decoding analyses are documented in the README. A controlled S2S sampling screen is pending. |
| Matched external IgLM | [fixed-length pilot](results/by_experiment/external_iglm_benchmark/fixed_length_pilot/scores/), [length-free pilot](results/by_experiment/external_iglm_benchmark/length_free_pilot/scores/) | Frozen 25-framework validation pilots, not a final held-out comparison. Scores need clean-tree regeneration before quotation. |
| Additional seeds | [early MLM check](results/by_experiment/additional_seeds/early_mlm_seed_check/), [dropout screen](results/by_experiment/additional_seeds/context_dropout_two_seeds/), [ESM-2 confirmation](results/by_experiment/additional_seeds/esm2_dropout_confirmation/) | MLM context pairs at seeds 42/43 and ESM-2 dropout seeds 44/45 are complete. Further selected seeds and fixed-seed repeat variability remain pending. |
| Light-chain loops | [validation OAS matrix](results/by_experiment/light_chain/validation_oas_matrix/batch_tables.md) | Historical runs only. Target-loop-specific splitting is needed before generalization claims. |
| Final frozen protocol/test | — | No final held-out experiment yet. |

The scratch/PLM learning-rate and loss screens are intentionally omitted from this category
view. See the corresponding `*sweep*` folders and the README findings for their results.

## Restoration status — 2026-10-06

The IgBert, ProtBert and ESM-2 caches are restored to the default lookup path via
a symlink to `/mnt/hdd/pang/hf-cache/hub`. Offline cache integrity, all five processed
dataset hashes, both environments on GPU 0, real ESM-2/ProtBert forward passes, a
saved S2S checkpoint with the real IgBert judge, and a packaged IgLM validation
smoke run all passed. No downloads or package changes were needed. See
[environment restoration](ENVIRONMENT.md#destination-server-restoration-2026-10-06)
and the [saved evidence](results/misc/restoration_20261006/report.json).

## Controls completed — checked 2026-10-07

All six trials in `20261006_140502_185951_no_partner_controls` have compatible
saved configs, checkpoints and generation results. The audit covers 900 framework
records and 18,000 candidates, with identical framework selection, 100% validity
and 100% judge coverage. **No rerun is needed.** Two blank AR rows in the generated
table are the original TensorBoard failures; `runs.tsv` points to their successful
`_2` retries. See the [six-trial review table](results/reviews/20261007-no-partner-controls.md).

All four trials in `20261007_013541_142543_esm2_dropout_confirmation` passed the
[full completion audit](results/reviews/20261007-esm2-confirmation.md): eight
scored outputs, 24,000/24,000 valid draws, full judge coverage, reproduced metrics
and final weights matching selected checkpoints. The recovered fourth trial
preserved early-stopping state. Missing-partner edit gains are +0.0838/+0.0616;
partner-present changes are −0.0042/−0.0039. **No rerun is needed for completion.**
Dirty-checkout and source/runtime provenance remain explicit; cluster-aware
uncertainty and final held-out evaluation are still pending.
Both [scratch MLM no-partner controls passed their audit](results/reviews/20261007-mlm-no-partner-controls.md):
6,000 returned nonempty canonical candidates, complete judge coverage, reproduced
summaries and selected weights/length-head artifacts checked. No rerun is needed.
MLM attempt validity is unrecorded.
All four partner-trained MLM dropout trials also passed their [completion audit](results/reviews/20261007-mlm-dropout-screen.md): eight evaluations, 24,000 returned candidates and complete judge coverage. Missing-partner edit gains are +0.2332/+0.2526, with small mixed partner-present changes (−0.0017/+0.0013). Dropout-0.5 absent scores are close to the no-partner-trained controls, without an equivalence claim. Source edits during training and the post-completion command refactor are explicitly audited. **No completion rerun is needed for any of these batches.** Historical MLM sweeps remain useful for their scoped exploratory claims.

The [OAS4 matched length-free benchmark](results/by_experiment/external_iglm_benchmark/oas4_length_free_validation/) is now audited complete: seven arms, 21,000 valid candidates and full judge coverage. It reuses the six scratch no-partner checkpoints and one frozen IgLM model; this is validation evidence, not a final test estimate.

## Next work order

1. **Reuse the completed matched validation benchmark.** The [seven-arm audit](results/reviews/20261008-benchmark-full.md) passed: identical 150-framework membership, 21,000 valid attempts and full heavy-only judge coverage. [§16](README.md#16-matched-oas4-length-free-validation-benchmark) gives the seven rows; the manuscript now includes validation and accounting tables. No completion rerun is needed. The earlier one-donor pilot and partner-present judge values remain separate.
2. **Reuse the completed empirical/paired-analysis follow-up.** Its [audit](results/reviews/20261008-benchmark-followup.md) passed: 3,000 new training-only draws, 24,000 valid/judge-scored candidates in total, unchanged original seven rows, all split labels verified, 13 primary fixed-pool contrasts and saved capacity/actual exposure. All learned-system versus empirical intervals are positive; local-checkpoint versus IgLM intervals span zero. The 150 frameworks have 150 distinct groups. No completion rerun is needed; training/generation variability and tuning exposure remain separate limitations.
3. **Reuse the completed local split-overlap audit and decide final scope.** [Audit](results/reviews/20261008-leakage-audit.md): 3/150 existing benchmark frameworks have ≥85% same-length training H3 neighbours; the test exclusion rule retains 14,818/15,273 records (97.02%). No completion rerun is needed. Choose the common final cohort, [external baseline coverage](benchmark/baseline_scope.md) and reporting scope, with [external pretraining exposure](benchmark/pretraining_overlap.md) stated as unverified. Framework documentation/release and methods review proceed alongside this; final held-out inference remains pending.
4. **Reuse the completed context evidence.** Eight no-partner controls, four ESM-2 confirmations and four MLM dropout trials need no completion rerun. Preserve the audited source/runtime differences. Additional shared seeds/repeats follow the precision needed for the paper's claims.
5. **Keep claim-dependent extensions lower priority.** Saved IgBert diagnostic checkpoints can be evaluated without training if conditioning is central. Same-architecture pretrained/random/frozen controls are needed for an isolated pretraining claim. Further LR, decoder and backbone sweeps remain optional.
6. **Freeze and run final held-out evaluation once.** Current validation values do not fill that table. Complete release documentation, authors/licences and formatting alongside this work.

Do not infer an architecture winner from unequal capacities, tuning exposure, length-matched populations or two-seed screening evidence. Experiment scripts remain user-run.
