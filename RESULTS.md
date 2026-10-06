# Experiment results by question

The category view at `results/by_experiment/` is generated from
[`results_catalog.json`](results_catalog.json) with
`/home/pang/miniconda3/envs/cdrprj/bin/python scripts/organize_results.py` (or any Python 3.11+).
It uses relative symlinks. The original timestamped folders in `results/runs/` remain the
source records: their saved `run_config.json`, `plan.json`, and internal paths are not rewritten.
A batch can appear in several categories. Hyperparameter sweeps are excluded from this view;
their summaries remain in `results/runs/*sweep*/`. This is an inventory, not a claim that
experiments within a category used identical protocols.

| Experiment type | Results to read | What the evidence covers |
| --- | --- | --- |
| Scratch baseline | [validation OAS matrix](results/by_experiment/scratch_baseline/validation_oas_matrix/batch_tables.md), [AR length-corrected](results/by_experiment/scratch_baseline/ar_length_corrected/batch_tables.md), [S2S pair](results/by_experiment/scratch_baseline/s2s_scratch_pair/batch_tables.md), [three-method comparison](results/by_experiment/scratch_baseline/three_method_comparison/batch_tables.md) | Completed exploratory MLM/AR/S2S runs. The earliest 07-24 and 07-29 matrices and pre-validation 08-17 matrix are also indexed, but selected on the test partition or predate the validation-safe OAS rerun. |
| Pretrained models | [S2S encoder matrix](results/by_experiment/pretrained_models/s2s_encoder_matrix/batch_tables.md), [validation OAS matrix](results/by_experiment/pretrained_models/validation_oas_matrix/batch_tables.md) | ESM-2/IgBert/ProtBert S2S encoders and older MLM ESM-2; no pretrained AR arm. These are system comparisons, not an isolated pretraining effect. |
| Partner context | [three-method comparison](results/by_experiment/partner_context/three_method_comparison/batch_tables.md), [S2S encoder matrix](results/by_experiment/partner_context/s2s_encoder_matrix/batch_tables.md), [context review](results/reviews/0915-0917-review.md) | With/without context examples exist; matched no-partner training controls for the recent dropout screen are still absent. |
| Context dropout | [AR/S2S screen](results/by_experiment/context_dropout/ar_s2s_screen/batch_tables.md), [review](results/reviews/0915-0917-review.md) | Two seeds for AR, scratch S2S and ESM-2 S2S, with both evaluation regimes. The earlier MLM pilot is indexed separately. |
| Pretrained vs random initialization | — | No same-architecture control yet. |
| Frozen encoder | [IgBert diagnostic](results/by_experiment/frozen_encoder/igbert_three_arm_diagnostic/batch_tables.md) | Three-epoch frozen/trainable diagnostic at one seed; generation from the saved checkpoints is pending. |
| Conditioning/collapse diagnostics | [collapse report](results/by_experiment/conditioning_diagnostics/igbert_collapse_report/REPORT.md), [diagnostic summary](results/by_experiment/conditioning_diagnostics/igbert_diagnostic_summary/REPORT.md) | One collapse and one same-seed non-reproduction; incidence and a preventive intervention are unknown. |
| Length-free generation | [three-method comparison](results/by_experiment/length_free_generation/three_method_comparison/batch_tables.md), [S2S encoder matrix](results/by_experiment/length_free_generation/s2s_encoder_matrix/batch_tables.md) | EOS/inferred-length results and length analysis. The assembled three-method comparison is explicitly marked as spliced in its `PROVENANCE.md`; use the fresh comparison for headline analysis. |
| Decoding/sampling | [validation OAS matrix](results/by_experiment/decoding_sampling/validation_oas_matrix/batch_tables.md) | Historical MLM decoding analyses are documented in the README. A controlled S2S sampling screen is pending. |
| Matched external IgLM | [fixed-length pilot](results/by_experiment/external_iglm_benchmark/fixed_length_pilot/scores/), [length-free pilot](results/by_experiment/external_iglm_benchmark/length_free_pilot/scores/) | Frozen 25-framework validation pilots, not a final held-out comparison. Scores need clean-tree regeneration before quotation. |
| Additional seeds | [early MLM check](results/by_experiment/additional_seeds/early_mlm_seed_check/), [dropout screen](results/by_experiment/additional_seeds/context_dropout_two_seeds/) | Early MLM seeds and two recent dropout seeds. Selected contrasts still need more runs, including identical-setting repeats for nondeterminism. |
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

## Next run order

1. **Freeze a validation protocol before more comparative training.** Specify example IDs,
   length policy, candidate/attempt budget, context for generator and judge, endpoints,
   checkpoint selection, and how failures/length-matched subsets are reported. Settle whether
   IgBert judge perplexity can be supporting only, given model-family and OAS overlap. Keep the
   final test split untouched. The protocol decision and leakage audit gate final claims.
2. **Evaluate the three saved IgBert diagnostic checkpoints.** This needs no training. Use the
   exact command in [the diagnostic recipe](configs/experiments/20260915_context_dropout.md#evaluate-the-completed-igbert-diagnostic-checkpoints-first), after the judge cache works.
3. **Run matched no-partner training controls.** The dropout-0 arms in the completed screen were
   trained *with* partner context. The missing controls are two scratch AR trials (seeds 42/43)
   and four S2S trials (scratch and ESM-2, seeds 42/43), trained and generated without partner
   context while the judge retains partner context. The [six-trial spec](configs/experiments/20261006_no_partner_controls.json)
   and [prospective protocol/commands](configs/experiments/20261006_h3_followup.md) are prepared,
   and the controls batch has started. Its initial AR attempts failed before training on
   a missing TensorBoard dependency, now corrected; the recipe includes the retry command.
   Do not resume September plans with changed inputs.
4. **Confirm one selected dropout contrast, then isolate ESM-2 pretraining.** The prepared
   [four-run confirmation spec](configs/experiments/20261006_esm2_dropout_confirmation.json)
   compares ESM-2 dropout 0/0.5 at seeds 44/45 in both generator context regimes. Additional
   seed 46 and fixed-seed repeats remain pending time and results. Compare trainable pretrained,
   randomly initialized, and frozen pretrained encoders separately with matched architecture
   and context policy before attributing a gain to pretraining.
5. **Complete the external comparison and held-out evaluation.** Audit OAS overlap with IgLM and
   PLM pretraining data, repair the fixed-length population mismatch, rescore the validation pilot
   on a clean tree, then freeze settings and run the final test once.

Do not infer an AR/S2S winner from the current one-seed comparisons: capacity, tuning exposure,
length-matched populations and run-to-run variance still differ. The [README coverage table](README.md#experiment-coverage-across-methods)
remains the status summary; this page gives each cell a concrete path and the next-run dependencies.
