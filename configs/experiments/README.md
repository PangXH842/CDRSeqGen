# Running experiments

Use `scripts/run.sh` for discovery, planning, execution and batch status. Recipe
names come from each top-level JSON's `name`; there is no separate registry to
keep in sync. Paths inside configs and relative command arguments resolve from
the repository root, even when the launcher is invoked elsewhere.

The next inference-only comparison uses the same launcher through
`run.sh benchmark plan|smoke|run|status|resume`; its scientific recipe lives in
[benchmark/recipes](../../benchmark/recipes/oas4_h3_length_free.md). It reuses completed
checkpoints and has a separate benchmark plan/output contract from training batches.

```bash
./scripts/run.sh list
./scripts/run.sh plan mlm_no_partner_controls --gpus 0
./scripts/run.sh run mlm_no_partner_controls --gpus 0
./scripts/run.sh status results/runs/<batch>
./scripts/run.sh resume results/runs/<batch> --gpus 0
```

`list` and `plan` create no batch or training files. `plan` validates the corpus
and shows resolved arms, training/evaluation context, length policy, candidate
budget and judge context. Add `--json` to print the full frozen-plan inputs.
`run` creates a timestamped batch unless `--output PATH` selects a new directory.
`status` reads `plan.json` and `runs.tsv`, checks artifacts for completed rows,
and reports whether the batch lock is active. It also accepts `--json`.

`resume` reads the batch's saved plan; it does not reconstruct settings from a
recipe that may have changed since launch. It verifies source/runtime identity,
corpus hashes and trial identities, then uses the existing executor. Completed
trials are skipped; unfinished training starts again from scratch. Continuing
an optimizer checkpoint is a separate recovery operation. The interrupted S2S
trial was recovered separately; all four confirmation trials are now complete.

## Recipes and evidence

Completion status belongs in [RESULTS.md](../../RESULTS.md) and each batch's
`runs.tsv`. This table describes the available definitions, not a launch queue.

| Name | Runs | Purpose | Recipe/details |
|---|---:|---|---|
| `mlm_no_partner_controls` | 2 | Scratch MLM trained/generated without its partner, seeds 42/43; completed | [spec](20261007_mlm_no_partner_controls.json), [protocol](20261007_mlm_context.md) |
| `mlm_dropout_screen` | 4 | Scratch MLM dropout 0/0.5, seeds 42/43, both contexts; completed and audited | [spec](20261007_mlm_dropout_screen.json), [protocol](20261007_mlm_context.md) |
| `esm2_dropout_confirmation` | 4 | S2S ESM-2 dropout 0/0.5, seeds 44/45; completed | [spec](20261006_esm2_dropout_confirmation.json), [protocol](20261006_h3_followup.md) |
| `no_partner_controls` | 6 | AR/scratch S2S/ESM-2 S2S no-partner controls, seeds 42/43; completed | [spec](20261006_no_partner_controls.json), [review](../../results/reviews/20261007-no-partner-controls.md) |
| `context_dropout` | 12 | September AR/S2S screen, both contexts; historical protocol | [spec](20260915_context_dropout.json), [protocol](20260915_context_dropout.md) |
| `igbert_diagnostic` | 3 | S2S IgBert trainable/frozen/lower-LR diagnostic | [spec](20260914_igbert_diagnostic.json), [details](20260915_context_dropout.md) |

The completed context recipes need no completion rerun. Their `run` commands create new experiments; use `status` to inspect the saved batches and the linked audits for reporting.

Built-in model/loop suites, such as `compare`, are also accepted as experiment
names. Use `list` to see them. For an explicit spec path:

```bash
./scripts/run.sh plan configs/experiments/20261007_mlm_no_partner_controls.json
```

## Shared settings and new recipes

Keep related experiments' common settings in one flat base config under
`bases/`. The two MLM stages share [bases/h3_mlm_scratch.json](bases/h3_mlm_scratch.json):
architecture, optimizer, checkpoint selection, learned length-head fitting,
sampling and judge policy are defined there once. The recipes contain context
choices and seed-specific trials. They resolve to the same six configs as before
the launcher redesign.

Merge order remains **base config → recipe overrides → trial overrides**.
There is no recursive inheritance. A new comparison needs one JSON recipe;
give it a unique `name`, a `base_config`, and named `trials`, then check it with
`run.sh plan NAME`. Add a Markdown protocol only when its interpretation needs
more explanation. Launched recipes and saved plans retain their provenance;
avoid changing a shared base during related batches.

For sweeps, use the existing preset and pass varying dimensions explicitly:

```bash
./scripts/run.sh plan sweep:lr --config configs/s2s/h3_s2s_scratch.json \
    --grid lr=1e-4,3e-4 --grid seed=42,43 --gpus 0,1
./scripts/run.sh run sweep:lr --config configs/s2s/h3_s2s_scratch.json \
    --grid lr=1e-4,3e-4 --grid seed=42,43 --gpus 0,1
```

Grid dimensions are `lr`, `warmup`, `weight_decay`, `epochs`, `hidden`, `layers`,
`loss`, `core_w`, `anchor_w`, and `seed`. Other settings come from the preset/base
config; use a controlled recipe when a comparison needs different policies.
`--data PATH` explicitly overrides the corpus for a new plan or run.

Named `plan`, `run` and `resume` commands reject inherited legacy experiment
overrides, such as `SEED`, `LRS`, `DRY_RUN` or `EXPERIMENT_DIR`, and show the exact
`unset` command needed. Ordinary runtime settings, including `HF_HUB_OFFLINE`,
remain available. `PYTHON=/path/to/python` selects the interpreter.

Existing GPU-first commands and `scripts/sweep.sh` continue to use the same
executor and environment overrides. Calling `run.sh` without arguments displays
help; an explicit GPU-first `run.sh 0` retains the old `all` suite behavior.
Historical folders without a frozen plan remain readable through the results
index and visualization tools.
