# Experiment results by question

[Back to README](../README.md) · [Detailed findings](FINDINGS.md) · [Current work](PROJECT_STATUS.md)

## Completed validation evidence

This inventory carries forward the audited October evidence; the documentation was
reorganized on **26 October 2026** without rerunning experiments. The later seed/retrieval
analyses extend the original seven-arm benchmark. Historical category entries below
retain their own populations and protocol caveats.

**IgLM adaptation pilot, audited November 3–6:** the [full GPU replay](../results/controls/iglm_replay_full_20261102.json)
matches 150/150 examples and 3,000/3,000 attempts. The two-model zero-shot batch, the four-trial
IgLM-S fine-tuning batch and the four-trial control batch completed with 6,000, 12,000 and 12,000 valid,
fully judge-scored candidates respectively. The declared validation-core criterion
selects **IgLM-S lr 3e-5, epoch 11** (core accuracy 0.407213; edit similarity 0.389715,
versus zero-shot IgLM-S 0.385881); see the [audit and per-LR table](../results/reviews/20261103-iglm-adaptation.md).
The [controls audit](../results/reviews/20261105-iglm-controls.md) adds random-init IgLM-S at three learning
rates (core accuracy 0.400739–0.406348, every run at the 20-epoch cap) and a scratch AR of the same size
(0.403734); pretrained minus random-init core accuracy at matched learning rates is +0.0065, +0.0015 and
−0.000005. Both [larger-IgLM trials](../results/reviews/20261106-iglm-large.md) are also complete:
6,000 valid, fully judge-scored candidates, matching selected checkpoint weights and independently
reproduced edit/length/score means. Validation core selects **IgLM lr 1e-5, epoch 4**
(core 0.409725, edit similarity 0.386728). No completion rerun is needed; only the prepared
selected-checkpoint GPU benchmark smoke remains pending in the
[adaptation protocol](../configs/experiments/20261102_iglm_adaptation.md).

These seed-42 pilot results use partner-present IgBert judging and remain separate
from the frozen external benchmark. Prior CPU parity checks used some test-split
records for implementation diagnostics; final comparative held-out evaluation
remains pending and must disclose that development exposure.

**Ready to reuse:** all benchmark comparisons below use the same 150 OAS4 validation frameworks and 20 attempts per framework and arm. Context studies retain their original generator/judge context and source/runtime provenance. Counts overlap across analyses; do not add them as independent experiments.

| Evidence | Completed | Results / verification |
|---|---|---|
| Scratch MLM / AR / S2S | Seven training seeds per method (42–48): **21 checkpoints** | [Original seeds 42/43](../results/benchmark/oas4_h3_length_free_20261007), [additional seeds](../results/benchmark/20261009_weekend_seeds), [combined analysis](../results/benchmark/20261012_143836_oas4_h3_variability/variability.md) |
| External generators | IgLM and IgCraft; IgCraft temperature 0.1 and 1.0 each have 3,000 valid, judge-scored candidates | [IgCraft comparison](../results/benchmark/20261009_183032_oas4_h3_igcraft_20261009_extension), [temperature sensitivity](../results/benchmark/20261009_weekend_igcraft_t1) |
| Training-only references | Unconditional H3 sampling plus three flank-keyed retrieval variants; 3,000 valid, judge-scored candidates per arm | [Empirical follow-up](../results/benchmark/oas4_h3_followup_20261008), [retrieval outputs](../results/benchmark/oas4_h3_retrieval_20261012), [retrieval contrasts](../results/benchmark/20261012_143807_oas4_h3_retrieval_20261012_extension/analysis.md) |
| Same-seed repeats | Two additional seed-42 training runs per method; three runs per method including the original | [Saved candidates and scores](../results/benchmark/20261009_weekend_repeats), [repeat analysis](../results/benchmark/20261012_143836_oas4_h3_variability/variability.md) |
| Capacity study | Three larger sizes × two seeds × three methods: **18 additional checkpoints**, compared with the small models | [Saved outputs](../results/benchmark/20261009_weekend_capacity), [capacity figure](../results/benchmark/20261012_143836_oas4_h3_variability/capacity.pdf) |
| Context controls and dropout | Ten no-partner controls (including two MLM ESM-2 additions), four scratch MLM dropout trials and four selected ESM-2 confirmation trials; September AR/S2S screen retained | [MLM controls](../results/reviews/20261007-mlm-no-partner-controls.md), [AR/S2S controls](../results/reviews/20261007-no-partner-controls.md), [MLM dropout](../results/reviews/20261007-mlm-dropout-screen.md), [ESM-2 confirmation](../results/reviews/20261007-esm2-confirmation.md) |
| Local split-overlap audit | All validation/test queries checked under the same-length ≥85% positional-identity rule; **3/150** benchmark frameworks affected; **14,818/15,273** candidate test records retained | [Audit](../results/reviews/20261008-leakage-audit.md), [saved summary](../results/benchmark/oas4_h3_leakage_20261008/summary.md) |
| Independent evidence verification | Saved report dated **20 October: 852 passed, zero failed, one information note** (stale manuscript PDF) | [Verification report](../../iAIM/verification/REPORT.md), [machine-readable checks](../../iAIM/verification/report.json) |

For the reported means, uncertainty and limitations, start with
[§17](FINDINGS.md#17-expanded-validation-evidence-seeds-retrieval-and-capacity).
The final held-out cohort and evaluation protocol remain to be frozen.

The [model specifications](MODELS.md#main-validation-benchmark-saved-architecture-settings)
record the saved benchmark architectures and sequence/head/system parameter counts
separately from the current builder defaults.

## Historical categories and source records

The category view at `results/by_experiment/` is generated from
[`results_catalog.json`](../results_catalog.json) with
`/home/pang/miniconda3/envs/cdrprj/bin/python scripts/organize_results.py` (or any Python 3.11+).
It uses relative symlinks. The original timestamped folders in `results/runs/` remain the
source records: their saved `run_config.json`, `plan.json`, and internal paths are not rewritten.
A batch can appear in several categories. Hyperparameter sweeps are excluded from this view;
their summaries remain in `results/runs/*sweep*/`. This is an inventory, not a claim that
experiments within a category used identical protocols.

The October 7 control and MLM dropout category links are available. The two IgBert report
folders (`results/misc/diagnostics/20260914_igbert` and
`results/misc/diagnostics/20260915_igbert_summary`) and the September reviews
(`results/misc/reviews/`) were part of the 21 September migration archive and are kept under
`results/misc/`, so the category view regenerates in full. The recovered reports keep their
original internal paths and relative links (`results/diagnostics/...`, `results/reviews/...`);
read them in place rather than editing them.

| Experiment type | Results to read | What the evidence covers |
| --- | --- | --- |
| Scratch baseline | [validation OAS matrix](../results/by_experiment/scratch_baseline/validation_oas_matrix/batch_tables.md), [AR length-corrected](../results/by_experiment/scratch_baseline/ar_length_corrected/batch_tables.md), [S2S pair](../results/by_experiment/scratch_baseline/s2s_scratch_pair/batch_tables.md), [three-method comparison](../results/by_experiment/scratch_baseline/three_method_comparison/batch_tables.md) | Completed exploratory MLM/AR/S2S runs. The earliest 07-24 and 07-29 matrices and pre-validation 08-17 matrix are also indexed, but selected on the test partition or predate the validation-safe OAS rerun. |
| Pretrained models | [S2S encoder matrix](../results/by_experiment/pretrained_models/s2s_encoder_matrix/batch_tables.md), [validation OAS matrix](../results/by_experiment/pretrained_models/validation_oas_matrix/batch_tables.md), [IgLM-S pilot audit](../results/reviews/20261103-iglm-adaptation.md), [larger-IgLM audit](../results/reviews/20261106-iglm-large.md) | ESM-2/IgBert/ProtBert S2S encoders, older MLM ESM-2 and both sizes of the pretrained IgLM AR pilot (zero-shot and fine-tuned, one seed). These are system comparisons; the isolated contrast is the row below. |
| Partner context | [three-method comparison](../results/by_experiment/partner_context/three_method_comparison/batch_tables.md), [S2S encoder matrix](../results/by_experiment/partner_context/s2s_encoder_matrix/batch_tables.md), [six AR/S2S no-partner controls](../results/reviews/20261007-no-partner-controls.md), [two MLM controls](../results/reviews/20261007-mlm-no-partner-controls.md), [context review](../results/misc/reviews/0915-0917-review.md) | Ten no-partner controls are audited: scratch/ESM-2 MLM, scratch AR and scratch/ESM-2 S2S at seeds 42/43. The [new MLM ESM-2 audit](../results/reviews/20261022-mlm-esm2-no-partner-controls.md) adds 6,000 returned candidates; four MLM controls now return 12,000 without an attempt-validity ledger. The new MLM dropout comparison completes six matched checkpoints and ten evaluations; [audit](../results/reviews/20261007-mlm-dropout-screen.md). The September AR/S2S screen uses earlier source/runtime versions. |
| Context dropout | [AR/S2S screen](../results/by_experiment/context_dropout/ar_s2s_screen/batch_tables.md), [October confirmation audit](../results/reviews/20261007-esm2-confirmation.md) | AR/S2S screen at seeds 42/43, four audited ESM-2 trials at seeds 44/45, and the [completed four-run MLM screen](../results/reviews/20261007-mlm-dropout-screen.md), each evaluated in both regimes. Historical and new batches retain their distinct source/runtime provenance. |
| Pretrained vs random initialization | [IgLM controls audit](../results/reviews/20261105-iglm-controls.md) | AR only: random-init IgLM-S at three learning rates and a scratch AR of the same size, one seed, validation. Every random-init run reached the 20-epoch cap; convergence is unestablished and the contrast remains exploratory. No MLM or S2S control yet. |
| Frozen encoder | [IgBert diagnostic](../results/by_experiment/frozen_encoder/igbert_three_arm_diagnostic/batch_tables.md) | Three-epoch frozen/trainable diagnostic at one seed; generation from the saved checkpoints is pending. |
| Conditioning/collapse diagnostics | [collapse report](../results/by_experiment/conditioning_diagnostics/igbert_collapse_report/REPORT.md), [diagnostic summary](../results/by_experiment/conditioning_diagnostics/igbert_diagnostic_summary/REPORT.md) | One collapse and one same-seed non-reproduction; incidence and a preventive intervention are unknown. |
| Length-free generation | [three-method comparison](../results/by_experiment/length_free_generation/three_method_comparison/batch_tables.md), [S2S encoder matrix](../results/by_experiment/length_free_generation/s2s_encoder_matrix/batch_tables.md) | EOS/inferred-length results and length analysis. The assembled three-method comparison is explicitly marked as spliced in its `PROVENANCE.md`; use the fresh comparison for headline analysis. |
| Decoding/sampling | [validation OAS matrix](../results/by_experiment/decoding_sampling/validation_oas_matrix/batch_tables.md), [IgCraft sensitivity](../results/benchmark/20261009_weekend_igcraft_t1) | Historical MLM decoding notes are in [FINDINGS.md](FINDINGS.md#open-problems). IgCraft temperature sensitivity is complete; a controlled S2S sampling screen remains optional. |
| Matched external generators | [original OAS4 benchmark](../results/benchmark/oas4_h3_length_free_20261007), [IgCraft extension](../results/benchmark/20261009_183032_oas4_h3_igcraft_20261009_extension) | Completed 150-framework validation comparisons with IgLM and IgCraft. Older 25-framework pilots remain historical and are superseded for current reporting. |
| Additional seeds | [seven-seed analysis](../results/benchmark/20261012_143836_oas4_h3_variability/variability.md), [same-seed repeats](../results/benchmark/20261009_weekend_repeats), [dropout screen](../results/by_experiment/additional_seeds/context_dropout_two_seeds), [ESM-2 confirmation](../results/by_experiment/additional_seeds/esm2_dropout_confirmation) | Seven scratch seeds per method and two additional seed-42 repeats per method are complete. Context/dropout studies retain their own smaller seed counts; generation-pool variability remains unmeasured. |
| Light-chain loops | [validation OAS matrix](../results/by_experiment/light_chain/validation_oas_matrix/batch_tables.md) | Historical runs only. Target-loop-specific splitting is needed before generalization claims. |
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
and the [saved evidence](../results/misc/restoration_20261006/report.json).

## Controls completed — checked 2026-10-07

All six trials in `20261006_140502_185951_no_partner_controls` have compatible
saved configs, checkpoints and generation results. The audit covers 900 framework
records and 18,000 candidates, with identical framework selection, 100% validity
and 100% judge coverage. **No rerun is needed.** Two blank AR rows in the generated
table are the original TensorBoard failures; `runs.tsv` points to their successful
`_2` retries. See the [six-trial review table](../results/reviews/20261007-no-partner-controls.md).

All four trials in `20261007_013541_142543_esm2_dropout_confirmation` passed the
[full completion audit](../results/reviews/20261007-esm2-confirmation.md): eight
scored outputs, 24,000/24,000 valid draws, full judge coverage, reproduced metrics
and final weights matching selected checkpoints. The recovered fourth trial
preserved early-stopping state. Missing-partner edit gains are +0.0838/+0.0616;
partner-present changes are −0.0042/−0.0039. **No rerun is needed for completion.**
Dirty-checkout and source/runtime provenance remain explicit; cluster-aware
uncertainty and final held-out evaluation are still pending.
Both [scratch MLM no-partner controls passed their audit](../results/reviews/20261007-mlm-no-partner-controls.md):
6,000 returned nonempty canonical candidates, complete judge coverage, reproduced
summaries and selected weights/length-head artifacts checked. No rerun is needed.
MLM attempt validity is unrecorded.
All four partner-trained MLM dropout trials also passed their [completion audit](../results/reviews/20261007-mlm-dropout-screen.md): eight evaluations, 24,000 returned candidates and complete judge coverage. Missing-partner edit gains are +0.2332/+0.2526, with small mixed partner-present changes (−0.0017/+0.0013). Dropout-0.5 absent scores are close to the no-partner-trained controls, without an equivalence claim. Source edits during training and the post-completion command refactor are explicitly audited. **No completion rerun is needed for any of these batches.** Historical MLM sweeps remain useful for their scoped exploratory claims.

The [OAS4 matched length-free benchmark](../results/by_experiment/external_iglm_benchmark/oas4_length_free_validation) is now audited complete: seven arms, 21,000 valid candidates and full judge coverage. It reuses the six scratch no-partner checkpoints and one frozen IgLM model; this is validation evidence, not a final test estimate.

## MLM ESM-2 controls completed — 22 October 2026

Both trials in `results/runs/20261021_155531_796405_mlm_esm2_no_partner_controls` passed the [completion audit](../results/reviews/20261022-mlm-esm2-no-partner-controls.md). Seeds 42/43 return 6,000 canonical candidates with full judge coverage; edit is 0.3891/0.3883 and length MAE 4.035/4.236. Both select epoch 5 and stop at epoch 8. The no-partner table now has ten trials. This is a pretrained-MLM integration demonstration; no attempt-validity rate, isolated pretraining advantage or ESM-2 MLM dropout effect is inferred. The judge uses partner context, separately from the heavy-only external benchmark. No completion rerun is needed.

## Next work order

1. **Reuse the completed matched validation benchmark.** The [seven-arm audit](../results/reviews/20261008-benchmark-full.md) passed: identical 150-framework membership, 21,000 valid attempts and full heavy-only judge coverage. [§16](FINDINGS.md#16-matched-oas4-length-free-validation-benchmark) gives the seven rows; the manuscript now includes validation and accounting tables. No completion rerun is needed. The earlier one-donor pilot and partner-present judge values remain separate.
2. **Reuse the completed empirical/paired-analysis follow-up.** Its [audit](../results/reviews/20261008-benchmark-followup.md) passed: 3,000 new training-only draws, 24,000 valid/judge-scored candidates in total, unchanged original seven rows, all split labels verified, 13 primary fixed-pool contrasts and saved capacity/actual exposure. All learned-system versus empirical intervals are positive; local-checkpoint versus IgLM intervals span zero. The 150 frameworks have 150 distinct groups. No completion rerun is needed; training/generation variability and tuning exposure remain separate limitations.
3. **Reuse the completed local split-overlap audit and decide final scope.** [Audit](../results/reviews/20261008-leakage-audit.md): 3/150 existing benchmark frameworks have ≥85% same-length training H3 neighbours; the test exclusion rule retains 14,818/15,273 records (97.02%). No completion rerun is needed. Choose the common final cohort, [external baseline coverage](../benchmark/baseline_scope.md) and reporting scope, with [external pretraining exposure](../benchmark/pretraining_overlap.md) stated as unverified. Framework documentation/release and methods review proceed alongside this; final held-out inference remains pending.
4. **Reuse the completed context evidence.** Ten no-partner controls, four ESM-2 confirmations and four MLM dropout trials need no completion rerun. Preserve the audited source/runtime differences. Additional shared seeds/repeats follow the precision needed for the paper's claims.
5. **Keep claim-dependent extensions lower priority.** Saved IgBert diagnostic checkpoints can be evaluated without training if conditioning is central. Same-architecture pretrained/random/frozen controls are needed for an isolated pretraining claim; the AR IgLM-S pilot is a first single-seed, non-converged contrast. Further LR, decoder and backbone sweeps remain optional.
6. **Freeze and run final held-out evaluation once.** Current validation values do not fill that table. Complete release documentation, authors/licences and formatting alongside this work.

Do not infer an architecture winner from unequal capacities, tuning exposure, length-matched populations or two-seed screening evidence. Experiment scripts remain user-run.
