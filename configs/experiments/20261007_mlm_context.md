# MLM partner-context controls — 2026-10-07

Prepared for the user to launch after the S2S ESM-2 confirmation, now complete.
No training or generation was launched during preparation. These two stages use
scratch MLM to complete method coverage without repeating the historical MLM
backbone, loss or decoding sweeps. ESM-2 MLM is a separate, optional extension.

Here, “no context” means no partner light chain and no antigen. The heavy-chain
flanks and its other CDRs remain available around the masked H3.

## Smallest useful next batch

[`20261007_mlm_no_partner_controls.json`](20261007_mlm_no_partner_controls.json)
trains two no-partner models at seeds 42/43. This fills the missing MLM rows in
the October no-partner control table. Each model fits its own length head and
writes one `generation_eval.json` without generator partner context.

If the paper also compares robustness to partner removal across all three
methods, follow with
[`20261007_mlm_dropout_screen.json`](20261007_mlm_dropout_screen.json).
It trains four partner-enabled models: dropout 0/0.5 at the same seeds 42/43.
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

No executable-source edits are needed. Keep the same code and environment
through both stages so the new MLM contrast is internally matched. Comparisons
with earlier AR/S2S screens retain their recorded source/runtime differences.

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

All four confirmation trials now pass the batch runner's completion checks.
Start fresh timestamped directories for the MLM stages;
do not reuse the frozen S2S confirmation directory.

```bash
cd /mnt/hdd/pang/CDRSeqGen
conda activate cdrprj
unset EXPERIMENT_DIR SWEEP_DIR SEED FREEZE_ENCODER ENCODER_INIT DATA_PATH DRY_RUN
HF_HUB_OFFLINE=1 ./scripts/run.sh plan mlm_no_partner_controls --gpus 0
HF_HUB_OFFLINE=1 ./scripts/run.sh run mlm_no_partner_controls --gpus 0
```

After stage 1 completes, run stage 2 if cross-method partner-removal robustness
is in scope:

```bash
HF_HUB_OFFLINE=1 ./scripts/run.sh plan mlm_dropout_screen --gpus 0
HF_HUB_OFFLINE=1 ./scripts/run.sh run mlm_dropout_screen --gpus 0
```

For stage 1, check two completed `runs.tsv` rows, two generation outputs and two
length-head/config pairs. For stage 2, check four completed rows, eight generation
outputs and four length-head/config pairs. Check framework identities and the
fixed judge context before combining either batch into a table.

Inspect a batch with `./scripts/run.sh status results/runs/<batch>` and resume
with `./scripts/run.sh resume results/runs/<batch> --gpus 0`. Resume reads the
frozen settings and requires unchanged code/runtime and corpus contents. The
launcher skips completed trials but restarts unfinished training from scratch;
checkpoint recovery is a separate operation, as in the current S2S recovery.
