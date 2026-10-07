# MLM partner-context controls — 2026-10-07

Stage 1 is complete: both no-partner controls passed the [completion audit](../../results/reviews/20261007-mlm-no-partner-controls.md).
Stage 2 is also complete: all four partner-trained MLM dropout trials passed the [completion audit](../../results/reviews/20261007-mlm-dropout-screen.md). No completion rerun is needed.
No training or generation was launched by the assistant. These two stages use
scratch MLM to complete method coverage without repeating the historical MLM
backbone, loss or decoding sweeps. ESM-2 MLM is a separate, optional extension.

Here, “no context” means no partner light chain and no antigen. The heavy-chain
flanks and its other CDRs remain available around the masked H3.

<a id="smallest-useful-next-batch"></a>

## Two-stage matched comparison

[`20261007_mlm_no_partner_controls.json`](20261007_mlm_no_partner_controls.json)
trained both no-partner models at seeds 42/43. Their audited outputs fill the MLM
rows in the October no-partner control table. Each model has its own fitted
length head and one scored `generation_eval.json` without generator partner context.
The batch is `results/runs/20261007_144639_381443_mlm_no_partner_controls`; no rerun is needed.
Edit similarities are 0.3880/0.3906, length match is 7.57%/6.43%, and conditional
core recovery is 0.2006/0.1950 (114/94 frameworks). All 6,000 returned candidates
are nonempty with full judge coverage; attempt-validity accounting is absent.

The completed matched MLM partner-context comparison uses
[`20261007_mlm_dropout_screen.json`](20261007_mlm_dropout_screen.json).
It trained four partner-enabled models: dropout 0/0.5 at the same seeds 42/43.
Each checkpoint is evaluated both with and without its partner, giving eight
generation files. Within each seed pair, only training dropout changes.

Both recipes now reference [bases/h3_mlm_scratch.json](bases/h3_mlm_scratch.json)
for their shared settings; their effective configs are unchanged. See the
[experiment guide](README.md) for the common launch interface and recipe list.

| Training policy | Generator with partner | Generator without partner | Models |
|---|---|---|---|
| No partner, dropout 0 | Outside this study | Yes | 2 |
| Partner, dropout 0 | Yes | Yes | 2 |
| Partner, dropout 0.5 | Yes | Yes | 2 |

Stage 1 is two training runs and 6,000 requested candidates. Both stages together
are six training runs and ten evaluations, with 30,000 requested candidates.
These counts include learned length-head fitting after each sequence model.
Two seeds give an initial matched comparison; they do not complete a final
training-stability study or replace held-out evaluation.

Compare no-partner training with always-partner training evaluated without its
partner to measure the cost of changing context between training and inference.
Compare dropout 0.5 with dropout 0 in each inference regime to measure robustness
and any partner-present cost. Keep per-seed results separate from uncertainty
over validation frameworks.

## Fixed protocol and checks

Both stages use OAS4 `data/processed/processed_oas_multi.json`, validation only,
SHA-256 `bd854ae32de886bc9b4466edf5d1aa3ade6c09847b622ceaa60cba395bbd5ad8`.
Keep the scratch architecture at width 128, four layers and eight heads, batch
32, LR `1e-4`, no warmup or weight decay, maximum assembled length 384, two-residue
anchors, up to 20 epochs and patience 3, selecting by masked core accuracy.
The LR is the existing cross-method scratch-control setting, not a new MLM
tuning study. Use full-span masking and plain cross-entropy in every arm.

Generation is length-free: `gen_length_mode=inferred` explicitly overrides the
MLM base config's native-length default. A checkpoint-specific classifier reads
a single gap sentinel, samples a length from classes observed in training, and
then independently decodes residues. Native length is used for scoring only.
The length head is fitted on the training dataset's same context policy,
including deterministic per-example partner dropout when enabled. Validation
and generation disable training dropout; ablation removes the partner explicitly.
Reuse the fitted head in both evaluation regimes, without fitting on validation
examples or fitting a replacement head for the partner-absent regime.

Use the same 150 uniformly selected validation frameworks (`gen_seed=42`) and 20
draws per framework, residue temperature 1, length temperature 1, top-p 0.9 and
top-k 0. The IgBert judge always receives the partner, independently of the
generator. Its perplexity remains a supporting endpoint because of OAS overlap.
MLM uses learned length classes rather than AR/S2S's EOS and 97-token decode cap;
record that method-specific length mechanism in comparisons.

Report framework-averaged edit similarity with candidate and framework coverage,
length-match rate, length MAE, diversity, conditional core/anchor recovery with
its contributing population, and judge coverage. Do not infer attempt validity
from the number of retained MLM samples if the output does not record attempts.

Historical MLM evidence remains useful for the appendix and exploratory plots.
The September 8 comparison has seed-42 no-partner and partner checkpoints, but
its judge context follows generator context. The July dropout records do not
yet establish this six-cell, two-seed, length-free protocol. Preserve those
results; do not relabel them as these new matched runs. The August backbone,
loss and decoding matrices need no rerun for their existing historical claims.

The batch froze source fingerprint `66adbd7ee4dd…` and the same runtime as the MLM controls/ESM-2 confirmation. During stage 2, MLM annotations changed and the identical loss-config construction/logging moved after model loading. All other executable AST nodes match the reconstructed frozen source. The command-interface refactor happened after completion. The audit retains archived source, unchanged plan/ledger hashes and the limitation that launch-time source identity does not prove byte-identical files at every trial start. Comparisons with earlier AR/S2S screens retain their own source/runtime differences.

**Observed contrast:** dropout improves partner-absent edit similarity by +0.2332/+0.2526 at seeds 42/43, restoring it close to the no-partner-trained controls (0.3907/0.3915 versus 0.3880/0.3906). Partner-present changes are −0.0017/+0.0013. These are descriptive validation effects, not significance or equivalence. Eight outputs contain 24,000 returned nonempty canonical candidates with full judge coverage; no attempt-validity rate is recorded. All six models across both stages have checked selected weights and fitted length heads.

Initial preparation passed 17 CPU-only spec/context tests, parsing all six MLM trials
and verifying that only the intended context policy and training seed change.
Both prospective plans validated the corpus hash and named splits. They are
saved under `results/reviews/20261007_mlm_*_plan.json`; the launcher creates its
own frozen plan when the user starts a batch. The executable source/runtime
identity matches the completed S2S confirmation plan.
The launcher redesign passed 35 focused CPU tests, including legacy commands,
named plan equivalence, explicit sweep grids, active-lock reporting, and frozen
resume with missing artifacts or changed recipes/data/source.

## Commands

Both stages are audited complete. These commands inspect the existing batches without starting experiments:

```bash
cd /mnt/hdd/pang/CDRSeqGen
conda activate cdrprj
./scripts/run.sh status results/runs/20261007_144639_381443_mlm_no_partner_controls
./scripts/run.sh status results/runs/20261007_163535_226387_mlm_dropout_screen
```

Stage 1 has two completed rows, two generation outputs and two length-head/config pairs. Stage 2 has four completed rows, eight generation outputs and four length-head/config pairs. No missing config needs rerunning.

The recipes remain available for intentional new, source-versioned experiments, not completion repair. Do not resume these completed batches after the command refactor: strict source identity remains enforced. Preserve their original plans and outputs.

Next priority is the [matched framework/external comparison](../../RESULTS.md#next-run-order), after freezing endpoints/populations, deciding leakage limitations and preparing uncertainty analysis. Additional MLM seeds/repeats are a targeted confirmation option if the dropout effect becomes a central claim. Training and generation remain user-run.
