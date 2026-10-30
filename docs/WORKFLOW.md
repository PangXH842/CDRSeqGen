<!-- Note to AI: Edits allowed OUTSIDE of MANUAL PROTECTION ZONES and INSIDE of AI ACCESS ZONES -->
[← back to README](../README.md)

*The operating manual: preprocessing, training, visualization, generation, evaluation and
benchmarking. Use the [experiment guide](../configs/experiments/README.md) for the current
named `list`, `plan`, `run`, `status` and `resume` interface.*

**Results and their caveats live in [FINDINGS.md](FINDINGS.md), with completed batches in
[RESULTS.md](RESULTS.md). Current work is tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).**

Some sections below preserve older commands and preparation-time status statements.
Completed experiments are not a launch queue. The appended S2S notes are historical;
the linked recipes and results index determine what is current.

---
# 🚀 Workflow & Execution

## Architecture Overview

![CDRSeqGen data, training, evaluation, analysis, and benchmark workflow](../misc/CDRSeqGen260814.drawio.png)

> The diagram is the 2026-08-14 snapshot. It predates the `cdrseqgen/methods/mlm/` consolidation and
> the `generate.py` / `evaluate.py` / `fit_mlm_length.py` renames; a refreshed diagram is pending.

## Step 1: Preprocessing Data
Convert datasets into a custom "Bio-Object" JSON format, extracting the regions and preserving `ANARCI` numbering.

**Data layout (2026-09-17):** Python dataset/preprocessing modules live in `cdrseqgen/datasets/`;
raw downloads and processed JSON/metadata live in root `data/raw/` and `data/processed/`.
The existing data was moved without reprocessing. This checkout retains the gitignored symlink
`dataset/data -> ../data`, so historical configs and absolute paths still resolve. Saved run
configs, manifests and result provenance retain their original paths. New commands/configs use
`data/`; preprocessing is invoked as `python -m cdrseqgen.datasets.preprocess.<source>`, or
equivalently `python -m cdrseqgen.datasets.preprocess --source <source>` -- both run the same
pipeline in `preprocess/__main__.py`.
On another checkout with historical runs, recreate the compatibility path with
`mkdir -p dataset` followed by `ln -s ../data dataset/data` if that path does not already exist.
A directory migration does not make old experiment plans resumable under changed source code;
start a new batch after this migration. Trial identity now uses corpus content rather than its
location, while the resolved path remains recorded in provenance.

**OAS**: the single-file example below builds the historical 21,106-pair corpus. The multi-file workflow below builds the current 152,727-pair, four-donor corpus.
```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
    --input_file data/raw/oas/1287169_1_Paired_All.csv \
    --output_file data/processed/processed_oas_data.json
```

For hyperparameter tuning, create a separate three-way file. Test clusters are allocated first using
the historical shuffle, so the established test membership is preserved; validation is then taken
from the former training partition:
```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
    --input_file data/raw/oas/1287169_1_Paired_All.csv \
    --output_file data/processed/processed_oas_tuning.json \
    --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
```

**Several GPUs**: `./scripts/run.sh 2,5,6 mlm_heavy ar_heavy` shards the round's runs across them round-robin,
one process per device. The runs in a batch are independent, so that is where the parallelism lives
— there is no DDP inside a single training. They share one batch folder, so the tables and
dashboards at the end cover the whole round. Each trial writes `<batch>/logs/<config>_<id>.log`;
multiple GPUs print status lines without interleaving training output. Sweeps use the same sharding. Round-robin also balances cost: a suite lists cheap
scratch runs first and ESM-2 runs after, so striding hands each device a mix rather than giving one
of them the whole slow tail. A single GPU behaves exactly as before, streaming to the terminal.

**Run a suite against a second corpus** with `DATA_PATH=<processed.json> ./scripts/run.sh <gpu>
<suite>`, the same override `sweep.sh` takes — no duplicate configs. The run name picks up that
corpus's `dataset_tag`, so `processed_oas_multi.json` produces `h3_OAS4_ar_scratch` and can never be
confused in a batch table with a single-donor `h3_OAS_ar_scratch`. Add a `DATASET_TAGS` entry
*before* the general `oas` one for any new corpus; first match wins.

The override applies to **every** config in the round, so it is refused when one of them needs the
antigen and the override corpus has none — `all` plus an OAS `DATA_PATH` exits rather than running
the twelve SAbDab2 configs against antibody-only data. Use `oas ar` (or `oas lcdr ar`) for an
OAS-wide round and run `antigen` separately on its own corpus.

`download_oas_paired.sh` downloads **and extracts** — it verifies each archive with `gunzip -t`
before unpacking (a truncated OAS CSV reads without complaint in pandas), keeps the `.gz` so a bad
extraction never costs another download, and skips work that is already done, so re-running it is
cheap.

**Several donors, one corpus.** `--input_file` takes any number of CSVs. They are read with their
own metadata rows, concatenated, and *then* split once:
```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
    --input_file data/raw/oas/*_Paired_All.csv \
    --output_file data/processed/processed_oas_multi.json \
    --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
```
**The split must be assigned once over the merged corpus, which is why merging happens here rather
than by concatenating finished JSONs.** `add_split_field` clusters heavy CDR3s and allocates whole
clusters; run per file, each file gets an independent clustering, and one clonal family can land in
`train` from donor A and in `test` from donor B — the merged file would look perfectly well-formed.

Every entry carries `source.subject`, so a donor-held-out split is possible later; the sidecar lists
`donors`, a per-file `sources` breakdown, and `longest_loop` per region — the causal layout reserves
a fixed number of answer positions (`dataset_ar.LOOP_LENGTH_RESERVE`) and refuses anything longer at
data-load time, so a new corpus reports that number where it is built rather than letting a training
run discover it. `--dedup exact` (the default) drops repeated
heavy+light amino-acid pairs across the whole corpus — public clones recur across donors, and with
`--split random` or `none` nothing else stops an identical pair straddling the holdout. Under
`--split cluster` identical pairs already share a cluster, so dedup changes the corpus size but not
its leakage properties; `--dedup none` reproduces pre-2026-09-03 behaviour. Clustering is O(n²):
~18 s at 21 k sequences, ~10 min at 120 k, ~3 h at 500 k.

**SAbDab2**: currently including 3984 structures - paired with resolved antigen, remaining >10k structures without antigen pending future inclusion
```bash
python -m cdrseqgen.datasets.preprocess.sabdab2 \
    --input_file data/raw/sabdab2/splits_final/abag_split.csv \
    --output_file data/processed/processed_sabdab2_data.json
```

Processed records preserve provenance needed by external baselines and future multi-source datasets:
`species`, both original IDs in `chain_sequence_ids`, and a stable `source` block (OAS run or
SAbDab/PDB identifiers). OAS species is read from its raw metadata row; the current SAbDab2 source
has an empty species column, so those records deliberately store `"species": null` rather than
assuming human. Each preprocessing run also writes a hash-verified `*.metadata.json` sidecar with
source details, filters, numbering convention, and split parameters.

## Step 2: Training & Batch Experimentation
Training settings come from JSON or CLI flags. The Trainer reports teacher-forced
residue recovery; generated-candidate evaluation is a separate optional stage.
Runs record logs, the epoch CSV and training curves. TensorBoard reporting is optional
and controlled by `TrainingArguments.report_to`.

The [argument reference](ARGUMENTS.md) lists shared and method-specific fields,
defaults and validation rules. Trailing flags override only the supplied fields:
for example, adding `--seed 7` preserves a JSON `"gen_judge": null` and empty lists.

> **⚠️ Every config must declare `eval_split`.** The split named here drives early stopping,
> `load_best_model_at_end`, and every reported `eval_*` number, so leaving it implicit means
> selecting checkpoints on whichever partition happens to be non-train — which, on a two-way file,
> is the final test set. `get_datasets` now **refuses** to start when `eval_split` is unset and the
> dataset carries a `test` split. The 36 OAS configs declare
> `processed_oas_tuning.json` + `"eval_split": "validation"`; the 12 SAbDab2 configs declare
> `"eval_split": "test"` because that dataset ships no validation partition (an open problem, not
> an endorsement). Use `test` only after the settings are frozen.

**Run a Single Experiment:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py configs/mlm/heavy/h3_scratch.json       # masked-LM
CUDA_VISIBLE_DEVICES=0 python train.py configs/ar/h3_ar_scratch.json    # autoregressive
```

> **One entrypoint, three methods.** `train.py` dispatches on the config's `"method"` key — absent
> means `mlm`. `--method ar` overrides the route; the config must use fields supported by
> that method's parser. Each method owns `main()` in
> `cdrseqgen/methods/<method>/train_cli.py`, which configures logging, parses arguments
> and connects `training.prepare_training()` / `finish_training()` to
> `common.training.run_training_pipeline()`. `train.py` routes to that command.
>
> All three methods use the same artifact names — `epoch_metrics.csv`, training-curve PNGs,
> `run_config.json`, `train.log` and `generation_eval.json` — for the stages enabled in
> the run. Shared plotters and table generators read those artifacts.
> Runs are tagged `h3_OAS_ar_scratch`, the `ar` marker keeping
> them distinct from the masked run of the same loop, dataset and context.
>
> **⚠️ Comparing the two is itself a matched-comparison problem.** The masked fixed-native track
> *hands the model the loop length* — 4.05 bits on H3 (see "How to read the numbers" in the README)
> — while a causal model must predict its own via `<eos>`. An AR-versus-MLM table built from
> masked fixed-native numbers flatters the causal side for a reason unrelated to the objective.
> Compare length-free on both sides: `evaluate.py --length_mode inferred` is the masked half.

`generate.py` also dispatches through the method registry. AR/S2S use length-free probe cases;
existing showcase cases that explicitly request `target_length` need a compatible fixture.

**One experiment interface:** [scripts/run.sh](../scripts/run.sh) selects the project
environment and routes named commands through [scripts/experiments.py](../scripts/experiments.py).
The existing [planner/executor](../cdrseqgen/experiments.py) owns dataset validation,
GPU scheduling, artifact checks, frozen plans and reports. Start with the
[experiment guide](../configs/experiments/README.md) for recipe names and shared settings:

```bash
./scripts/run.sh list
./scripts/run.sh plan mlm_no_partner_controls --gpus 0
./scripts/run.sh run mlm_no_partner_controls --gpus 0
./scripts/run.sh status results/runs/<batch>
./scripts/run.sh resume results/runs/<batch> --gpus 0
```

Named commands take execution choices as flags: `--gpus`, `--output` for a new
batch, and `--data` for an explicit corpus override. Scientific settings stay in
the recipe/base config; related recipes can share a flat base under
`configs/experiments/bases/`. Planning writes no files; `--json` prints every merged
config and the corpus/source identity. The no-argument launcher displays help.

GPU-first commands and [scripts/sweep.sh](../scripts/sweep.sh) remain compatible:

```bash
./scripts/run.sh 0 compare
./scripts/run.sh 0,1 sweep lr configs/s2s/h3_s2s_scratch.json
./scripts/sweep.sh 0,1 lr configs/s2s/h3_s2s_scratch.json  # identical sweep
DRY_RUN=1 ./scripts/run.sh 0 compare                  # inspect merged configs; write nothing
```

Both modes share dataset validation, GPU sharding, per-trial logs, completion checks, and resume.
`DATA_PATH` overrides a corpus only when explicitly set; otherwise each base config's path wins.
The launchers work from any directory, resolving relative config/data/output paths from the project
root. Set `PYTHON=/path/to/python` to select an interpreter; otherwise they activate `cdrprj` when
the usual Conda installation exists, falling back to the active Python elsewhere.

Each executed round saves `plan.json` (merged configs, dataset hashes, source/runtime identity,
and ranking policy), `.configs/` (the frozen inputs passed to training), and `runs.tsv` (one current
status per trial, with device, log, and run paths). Training still writes its fully resolved defaults
to each run's `run_config.json`. This unifies execution; it does not automatically promote a sweep
winner into the shipped training configs.

Use `run.sh resume BATCH --gpus 0` to resume from the frozen plan, checking source,
runtime, corpus content and trial identities without rereading a changed recipe.
It skips completed trials with compatible artifacts and restarts unfinished
training from scratch. Optimizer-checkpoint recovery remains a separate operation.
The legacy commands use `EXPERIMENT_DIR=results/runs/my_round`; `SWEEP_DIR` remains
an alias in sweep mode. Those commands rebuild the requested plan and refuse
changed configs, data, source or protocol.
Older folders without `plan.json` remain available for analysis; start a fresh folder for this
runner. A retry updates its trial's status rather than counting as another training seed.

**Context-dropout experiments:** training dropout is disabled in validation for all methods.
`gen_eval_context_ablation=true` evaluates the selected checkpoint with and without partner context,
keeping the judge's partner context fixed. It requires a partner-enabled config and generative
evaluation. Both `generation_eval.json` and `generation_eval_without_partner.json` are required for
completion; the batch table includes both regimes. [Screen protocol and checkpoint-evaluation commands](../configs/experiments/20260915_context_dropout.md).
For manual evaluation use the original config with `--partner_context with|without`; use
`--judge_partner_context with|without` to explicitly hold judge context constant. Both default to
`trained` (the original config's context). Context choices are recorded in the output metadata and
filename identity. The test split remains unused during this screen.

**Run an experiment spec:** for a one-off controlled comparison, keep the recipe in a JSON file
under `configs/experiments/` instead of adding a permanent suite or another shell script:

```bash
./scripts/run.sh plan igbert_diagnostic --gpus 0
./scripts/run.sh run igbert_diagnostic --gpus 0,1,2
```

A spec contains `name`, an optional shared `base_config`, optional shared `overrides`, and a
nonempty `trials` list. Each trial has a unique `name`, optional `base_config`, and optional
`overrides`. Settings merge in this order: base config → shared overrides → trial overrides.
Config and data paths are relative to the repository root, as with regular suites. Trial names
use letters, digits, underscores, and hyphens; the first character must be a letter or digit.
Identical resolved trials are rejected; declare distinct settings or seeds explicitly.

Use `--data` with named commands; legacy commands accept `DATA_PATH`. Put
`seed`/`data_seed`, `encoder_init`, and `freeze_encoder` in the spec. Named commands
reject legacy experiment environment overrides so settings cannot change invisibly.
For GPU-first commands, `DRY_RUN`, GPU assignment, `EXPERIMENT_DIR`, resolved `plan.json`,
completion checks, and batch reporting use the existing suite executor. The dated IgBert recipe
can be archived after diagnosis without changing the runner; its monitoring and scheduler-preserving
stop options remain reusable.

**Run an Automated Experiment Suite:** [`scripts/run.sh`](../scripts/run.sh) takes a GPU id and one or more suite names, skips to the next config if one fails, and writes the round's tables and dashboards into the batch folder when it finishes (Step 3).
```bash
# One runner, suites chosen (and combined) by arg:  ./scripts/run.sh <gpu> [suite ...]
# Every suite names its METHOD first, then what varies within it.
./scripts/run.sh 4                     # default "all": mlm_heavy + mlm_antigen + mlm_light (36)
./scripts/run.sh 4 mlm_heavy           # masked OAS heavy, scratch+ESM-2 (h1/h2/h3)         (12)
./scripts/run.sh 4 mlm_antigen         # masked SAbDab2 antigen matrix (_ag / _ag_lconly)   (12)
./scripts/run.sh 4 mlm_light           # masked OAS light (l1/l2/l3, heavy-chain context)   (12)
./scripts/run.sh 4 mlm_heavy mlm_light # combine suites — deduped, run in order             (24)

# Autoregressive, and the paired comparison
./scripts/run.sh 4 ar_heavy            # causal OAS heavy (h1/h2/h3), base + _lc              (6)
./scripts/run.sh 4 s2s_heavy           # scratch encoder-decoder H3, base + partner          (2)
./scripts/run.sh 4 s2s_esm2            # ESM-2 encoder + new decoder, H3 base + partner       (2)
./scripts/run.sh 4 s2s_igbert          # IgBert encoder + new decoder, same two contexts      (2)
./scripts/run.sh 4 s2s_protbert        # ProtBert encoder + new decoder, same two contexts    (2)
./scripts/run.sh 4 s2s_plm             # all three pretrained encoder families               (6)
./scripts/run.sh 4 compare             # MLM/AR/S2S H3, each base + partner                   (6)

# Antibody/protein-BERT PLM backbones on OAS heavy (first run downloads a ~1.6GB model each)
./scripts/run.sh 4 mlm_igbert          # IgBert (antibody-specific)                          (6)
./scripts/run.sh 4 mlm_protbert        # ProtBert (general-protein baseline)                 (6)
./scripts/run.sh 4 mlm_plm             # both (mlm_igbert + mlm_protbert)                   (12)
```

**S2S pretrained encoders.** Start with `s2s_esm2`. The configs under
`configs/s2s/h3_s2s_{esm2,igbert,protbert}[_lc].json` train a pretrained bidirectional encoder plus
our newly initialized residue decoder. These are encoder-pretrained models; the decoder has no
pretrained weights. The encoder reads its own amino-acid ids, one native mask token, and native
CLS/end wrappers. The decoder keeps the same 27-token vocabulary and EOS as scratch S2S.
A learned projection connects the encoder width to the decoder width.

The initial configs use H3/OAS4 validation, max_len 384, a 128-wide two-layer decoder, LR 3e-5,
0.06 warmup, 0.01 weight decay, and ten epochs with early stopping. Batch size 4 with eight
accumulation steps gives an effective training batch of 32 on one GPU. These are starting settings,
not fitted hyperparameters. The generation protocol matches the H3 comparison: 150 frameworks,
20 candidates, top_p 0.9, seed 42, IgBert judge, and EOS-selected length.

```bash
DRY_RUN=1 ./scripts/run.sh 0 s2s_esm2       # review two runs without loading weights
./scripts/run.sh 0 s2s_esm2                # fine-tune ESM-2 + train the decoder
FREEZE_ENCODER=true ./scripts/run.sh 0 s2s_esm2  # fixed encoder features; train projection/decoder
ENCODER_INIT=random ./scripts/run.sh 0 s2s_esm2  # same architecture with a random encoder

# Both controls work in sweep mode too. Keep each setting in its own experiment directory.
LRS="1e-5 3e-5 1e-4" EPOCH_LIST="10" ./scripts/sweep.sh 0 lr configs/s2s/h3_s2s_esm2.json
# The corresponding config/CLI fields are encoder_init and freeze_encoder:
CUDA_VISIBLE_DEVICES=0 python train.py configs/s2s/h3_s2s_esm2.json --freeze_encoder true
```

`hidden_size`, `num_attention_heads`, `intermediate_size`, and `num_decoder_layers` configure
the new decoder; `hidden_size / num_attention_heads` must be a whole, even number (rotary positions),
which `train.py` checks before it creates the run directory.
Leave `num_encoder_layers` unset: encoder dimensions come from the PLM checkpoint.
Frozen encoders keep dropout off during training. Random initialization retains the same encoder
architecture, tokenizer, and parameter count, so it can help separate pretraining from capacity.
Encoder controls and their differences are included in the shared runner's plan/resume identity.

Model checkpoints contain the full encoder configuration and weights, projection, and decoder.
Final and intermediate training checkpoints also save the encoder tokenizer. `generate.py` and
`evaluate.py` prefer those saved tokenizer files and use the existing S2S generation/evaluation
path. Reports distinguish `s2s_esm2` from masked ESM-2 and include an S2S PLM recovery table.

A full pretrained encoder-decoder such as ProtT5 is a separate future adapter. Its released
checkpoint used BART-like residue masking, not original T5 span denoising; its architecture alone
does not establish a native missing-CDR generation format
([official model card](https://huggingface.co/Rostlab/prot_t5_xl_uniref50)).

The `compare` suite runs MLM, AR, and S2S in one batch, each with and without partner context.
All six receive `max_len=384`, temperature 1, top-p 0.9, top-k 0, 150 frameworks × 20 samples,
generation seed 42, and the IgBert judge. MLM uses inferred length; AR/S2S terminate with EOS.
Training settings remain those declared in the configs. Parameter counts differ: S2S pins separate
encoder/decoder depths, so record those counts when interpreting the comparison.

The generation panel compares length-free outputs. Teacher-forced recovery remains a training
diagnostic: AR/S2S see preceding native residues, and MLM receives the native mask count.
Length-matched generation recovery also needs its contributing framework population reported.

**Every run lives at `results/runs/<batch>/<run>/` — always two levels under `runs/`.** A `run.sh` round shares one batch
folder; a standalone `train.py` gets a batch folder of its own, so `results/` never collects loose run
dirs. **`results/runs/latest`** is a symlink to the most recent *batch*, so `results/runs/latest/<config>/`
addresses any run in that round:

```text
results/runs/
├── 20260720_191138_all/                ← one ./scripts/run.sh round, tagged with its suites
│   ├── h1_OAS_scratch/
│   ├── h3_OAS_LC_scratch/
│   ├── batch_tables.md                 ← the round's own report, written when it finishes
│   └── figures/                        ← batch_oas.png, batch_sd2.png
├── 20260720_193002/                    ← a single train.py run
│   └── h3_OAS_scratch/
├── 20260811_181946_sweep_h3/           ← validation-safe sweep; results/runs/latest unchanged
│   ├── h3_OAS_scratch_lr1e-4_wu0.0_wd0.0_s42/
│   ├── h3_OAS_scratch_lr1e-3_wu0.06_wd0.01_s43/
│   ├── plan.json  runs.tsv  summary.tsv ← the records you read
│   ├── figures/                        ← objective_comparison.png
│   └── logs/                           ← one stdout/stderr capture per trial
├── 20260903_113933_ar/                 ← ./scripts/run.sh 3 ar  (suites join with _: _oas_lcdr)
└── latest -> 20260720_193002/          ← newest batch; generate.py uses latest/<config>
```

Everything a round produces about *itself* sits at its own root: records as files, charts under
`figures/`, per-trial stdout under `logs/`. The logs are kept rather than dropped because a trial
that dies before logging attaches leaves no `train.log` — sometimes no run directory at all — so
that capture is the only evidence of the failure; they just no longer outnumber the records.

A named group is `<timestamp>_<name>` — the timestamp leads so directories sort chronologically
whether or not they carry a name (`20260730_160028_seedcheck`, `20260811_181946_sweep_h3`).
`run.sh` names a round from the suites it ran (`_ar`, `_oas_lcdr`, `_all` for the default), so a
batch says what it contains without opening a `run_config.json`; batches predating 2026-09-03 are
bare timestamps and still parse, since nothing reads meaning from the batch name.

The run folder is named from the **resolved arguments**, not the config filename — `h3_scratch.json
--target_region l3` produces an `l3_OAS_HC_scratch` folder, so an override can't be misfiled. The
date/time lives on the batch folder and isn't repeated inside.

**Tag a run** with `CDRSEQGEN_RUN_SUFFIX` to distinguish runs that resolve to the same name:
```bash
CDRSEQGEN_RUN_SUFFIX=ablation_noLC python train.py configs/mlm/heavy/h3_scratch.json
#  -> results/<batch>/h3_OAS_scratch_ablation_noLC/
```
`sweep.sh` tags every trial with its LR, warm-up, weight decay, epochs and seed, so folders are
self-describing. It groups trials with `CDRSEQGEN_RESULTS_ROOT`, writes `runs.tsv`/`summary.tsv`, and
sets `CDRSEQGEN_SKIP_LATEST=1` so exploratory trials do not replace the main usable batch. If two runs
still collide, the second becomes `..._2` with a logged warning; a run never silently overwrites another.

A standalone `python train.py …` sets neither environment variable and retains the normal behavior.
Export `CDRSEQGEN_RESULTS_ROOT` yourself to group manual runs:
```bash
CDRSEQGEN_RESULTS_ROOT=results/runs/my_experiment python train.py configs/mlm/heavy/h3_scratch.json
```

**Override a config from the CLI:** trailing flags override the JSON (they do **not** replace it), so any
`TrainingArguments` field is tunable without writing a new config:
```bash
python train.py configs/mlm/heavy/h3_scratch.json --learning_rate 3e-4 --num_train_epochs 20 --warmup_ratio 0.06
```

The original `masking_policy: "full_span"` remains the default: every native CDR residue is replaced
by its own mask token. Opt into mixed-mask training when training a checkpoint for iterative
decoding:

```bash
python train.py configs/mlm/heavy/h3_scratch.json \
    --masking_policy mixed --mixed_full_span_probability 0.5
```

Mixed mode keeps the requested fraction of training examples fully masked. For every other example,
it reveals at least one native loop residue and masks at least one, with loss computed only over the
masked subset. Validation, post-training generation, and standalone evaluation always use a separate
`full_span` dataset view, so partially revealed native residues cannot leak into reported recovery.
Both settings are persisted in `run_config.json`.

**Choose a training loss:** ordinary token-averaged cross-entropy remains the default and therefore
reproduces old configs. The shared loss registry in
[`cdrseqgen/methods/mlm/losses.py`](../cdrseqgen/methods/mlm/losses.py) is applied after model forward,
so scratch, ESM-2, IgBert, and ProtBert use the same implementation. The first alternative raises
the contribution of the hypervariable core relative to the easier loop anchors:

```bash
python train.py configs/mlm/heavy/h3_scratch.json \
    --loss_type weighted_cross_entropy \
    --core_loss_weight 2.0 --anchor_loss_weight 1.0 --anchor_trim 2
```

or, equivalently, from a config: [`configs/mlm/heavy/h3_scratch_weighted.json`](../configs/mlm/heavy/h3_scratch_weighted.json)
is `h3_scratch.json` plus the three loss keys and nothing else, so the two differ only in objective.

The weighted objective is a weighted mean over supervised tokens, so only the core-to-anchor weight
ratio matters. `anchor_trim: 2` means the first and last two residues are anchors; loops of length
four or less contain no core and are treated entirely as anchors — on H3 that is 1 of 21,106 loops,
but on L2 it is 98.8% of them, so the weighted objective is an H3/H2/H1 tool and degenerates to plain
cross-entropy on the short light-chain loops. Setting `anchor_loss_weight 0` supervises the core only
(both weights zero is refused, as is a negative weight); a batch whose supervised tokens are all
anchors under that setting raises rather than emitting NaN. The same trim is passed to the
reported `core_accuracy`/`anchor_accuracy`, and original loop offsets keep the classification correct
under mixed masking. Loss settings and any `label_smoothing_factor` are persisted in
`run_config.json`.

**Tune on validation:** `run.sh <gpus> sweep [preset] [base_config]` runs a validation-only,
multi-seed screen. The older `sweep.sh <gpus> [preset] [base_config]` command forwards to it:

| preset | grid | trials |
|---|---|---:|
| `h3` (default) | 4 LR × 2 warm-up × 2 weight decay × 2 seeds, fixed 30-epoch horizon | 32 |
| `lr` | 3 LR × 2 seeds, warmup 0.06 / decay 0.01, 20 epochs | 6 |
| `pilot` | 3 LR × 2 epoch budgets — the 2026-07 "is the budget enough?" grid | 6 |
| `arch` | 3 widths × 2 depths, FFN tracking width | 6 |
| `loss` | 3 **distinct** objectives × 3 seeds, generative eval on, ranked by judge perplexity | 9 |
| `quick` | 2 trials at 2 epochs — checks the script runs, not for results |  2 |

The `loss` grid is `cross_entropy` plus four weighted settings (core/anchor `1.0/1.0`, `1.0/0.0`,
`3.0/1.0`, `3.0/0.0`) — but only **three** of those five are distinct objectives, so 24 nominal grid
points collapse to **9** trials.

> **Why five settings are three objectives.** `weighted_cross_entropy` is a weighted *mean* over
> supervised tokens, so the normaliser cancels any common factor and only the core-to-anchor
> **ratio** survives. Hence `weighted 1.0/1.0` is bit-identical to `cross_entropy` (verified to
> 1e-9) and `3.0/0.0` to `1.0/0.0`. The 08-25 screen ran all five before this was noticed, which is
> how Findings §13 ended up with the same loss occupying its top two ranks — a free measurement of
> run-to-run nondeterminism, and the reason §13 can say the objective differences are inside it.
> The planner canonicalises equivalent objectives and duplicate seeds before launching.
> For a deliberate determinism control, use separate experiment folders and label it as such.

```bash
./scripts/sweep.sh 4                      # h3 preset
./scripts/sweep.sh 4 pilot
SEEDS="42 43 44" LRS="3e-4 1e-3" ./scripts/sweep.sh 4
EXPERIMENT_DIR=results/runs/my_sweep ./scripts/run.sh 4 sweep h3  # exact-plan resume
./scripts/sweep.sh 4 loss
DRY_RUN=1 ./scripts/run.sh 4 sweep loss  # print merged trial configs, train nothing
```

Every dimension (`LRS`, `WARMUPS`, `WEIGHT_DECAYS`, `EPOCH_LIST`, `SEEDS`, `HIDDEN_SIZES`,
`NUM_LAYERS`, `PATIENCE`, `LOSS_TYPES`, `CORE_WEIGHTS`, `ANCHOR_WEIGHTS`) is
env-overridable, so a preset only supplies defaults. Four knobs control what each trial is scored on:
`RANK_METRIC` (`core` | `whole` | `judge_ppl` | `diversity` | `gen_core` | `length_mae` |
`length_match` | `valid_rate`), `SELECT_METRIC`
(`core_accuracy` | `accuracy`), `GEN_EVAL` (`true`/`false`), and `GEN_EXAMPLES`/`GEN_SAMPLES`, which
shrink the generative eval — it runs once per trial, so the 150×20 default would dominate a 9-trial
sweep. The `loss` preset uses 100×10.

**Sweeping a causal config** needs a free-running `RANK_METRIC`. `sweep.sh` is already
method-neutral — it passes the base config to `train.py`, which dispatches on the config's `method`
key, so `./scripts/sweep.sh <gpu> <preset> configs/ar/h3_ar_scratch.json` runs unmodified. But the
defaults (`core`, and `SELECT_METRIC=core_accuracy`) are **teacher-forced**, and for a causal model
teacher-forcing supplies the true loop prefix: a checkpoint can score 0.400 there while matching the
native length 8% of the time free-running (2026-09-02). Use `length_match` for exact completed,
non-empty lengths per **all attempts**, `length_mae` for cutoff-penalized attempt-level error, and
`valid_rate` to screen EOS/empty failures directly. `gen_core` and `judge_ppl` are the other
free-running choices; all require `GEN_EVAL=true`.

**`SELECT_METRIC` picks each trial's best checkpoint**, and it carries the same circularity as
ranking: `core_accuracy` is what a core-weighted objective optimises, so selecting on it advantages
those settings per-epoch even when the final ranking is judged independently. The `loss` preset
therefore defaults it to whole-span `accuracy`; every other preset keeps `core_accuracy`. Note
`eval_loss` is **not** an option — it is computed with each trial's own loss function, so its scale
differs between objectives and cannot be compared across them.

Rank by `judge_ppl` rather than `core` when comparing objectives: a loss that reweights core tokens
changes the very quantity `core` measures, so ranking on it partly rewards the objective for
optimising its own metric. The independent judge does not share that bias.

The script refuses a dataset without explicit `train`/`validation`/`test` labels, forces
`--eval_split validation`, selects each run's checkpoint on `SELECT_METRIC`, and ranks settings
across seeds on `RANK_METRIC`. It writes `runs.tsv`, `summary.tsv` and
`objective_comparison.png`, never selects the test partition, skips completed grid
points on resume, and sets `CDRSEQGEN_SKIP_LATEST=1` so exploratory trials do not replace the main
usable batch. It also flags a winner whose margin falls below the ~0.006 seed-noise floor as a tie.

Width and depth overrides are independent: `NUM_LAYERS="2 4"` works without `HIDDEN_SIZES`.
For S2S, a depth override sets both encoder and decoder depth; a width override leaves their
configured depths intact. Architecture sweeps require scratch backbones; weighted-loss sweeps
currently require MLM. Incompatible combinations fail before any trial starts.

`summary.tsv` records both `selection_metric` (checkpoint choice) and `rank_metric`/`score_metric`
(setting ranking), along with ranking direction. Seed counts refer to distinct training seeds.
Plots for new plans group by the full setting identity, so an LR sweep cannot pool different LRs
into one cross-entropy curve.

**On the `pilot` preset.** It sweeps `num_train_epochs`, which also reshapes the linear scheduler —
so a "20 epochs is better" result from it confounds budget with schedule. That is why `h3` fixes the
horizon and does not sweep the budget. `pilot` answers a different question (is the budget itself
enough?) and should not be used to select hyperparameters.

**Completed screen (2026-08-11).** All 32 trials completed on the three-way clustered OAS split
(16 settings × seeds 42/43). The leading region was shallow rather than decisive:

| LR | warm-up | weight decay | mean validation core recovery | seed SD | mean best epoch |
|---:|---:|---:|---:|---:|---:|
| **1e-3** | **0.06** | **0.01** | **0.32305** | 0.00138 | 16.5 |
| 3e-4 | 0.06 | 0.01 | 0.32269 | 0.00070 | 18.5 |
| 3e-4 | 0.00 | 0.00 | 0.32238 | 0.00032 | 13.5 |

The top-two gap is only 0.00036 and changes sign between the two seeds, so this screen identifies a
region, not a conclusive winner. Confirm the top two with additional shared seeds before freezing a
training config; evaluate the final choice only once on `test`. Full local records are in
`results/runs/20260811_181946_sweep_h3/{runs,summary}.tsv`.

## Step 3: Result Visualization & Comparison
All three recovery metrics are recorded **every epoch** (`compute_metrics` → `epoch_metrics.csv`:
`eval_accuracy`, `eval_core_accuracy`, `eval_anchor_accuracy`), and each run auto-generates four PNGs:

| file | shows |
|---|---|
| `eval_loss_curve.png` | loss |
| `eval_accuracy_curve.png` | whole-span recovery (anchor-inflated) |
| `eval_core_accuracy_curve.png` | **core** recovery — the one to report |
| `recovery_breakdown.png` | all three together — the anchor gap at a glance (H3: ~0.46 / ~0.32 / ~0.90) |

Use the **breakdown** chart to see the anchor gap, and the **individual** charts to read a trend: the
breakdown's y-axis spans ~0.3–0.9, which compresses within-line movement (ESM-2's core decay from
0.321 → 0.306 is legible on `eval_core_accuracy_curve.png`, barely visible on the breakdown).

### A finished round reports itself
What each run does for itself, a batch now does for the whole round — the report lands in the **batch
folder**, beside the runs it describes, so nothing has to be regenerated by hand to read a result.
Records sit at the batch root and charts under `figures/`, the same split `results/` uses one level up:

| written by | at the end of | into |
|---|---|---|
| `batch_tables.py` | `run.sh` | `<batch>/batch_tables.md` — every README comparison table |
| `batch_plots.py` | `run.sh` | `<batch>/figures/batch_<ds>.png` — one dashboard per dataset |
| `sweep_tables.py` | `sweep.sh` | `<sweep>/summary.tsv` — settings ranked across seeds |
| `sweep_plots.py` | `sweep.sh` | `<sweep>/figures/objective_comparison.png` — settings × seeds |

Both steps are guarded in the shell: a batch that already trained cannot be spoiled by a reporting
failure, and any of the four can be re-run against a finished directory at any time. That matters for
`sweep_tables` in particular — it used to be a heredoc at the end of `sweep.sh`, so an interrupted
sweep produced no summary at all, even though every finished trial was already on disk.

`results/figures/` stays for what does **not** belong to one batch: hand-picked charts for this
README, and ad-hoc comparisons spanning several batches (the two CLIs below still default there).

One function draws both comparison shapes, because every line is the same triple `(run_dir, column, label)`.
Each `--run` takes `LABEL FOLDER [METRIC]`; omit `METRIC` and it falls back to the global `--metric`
(default `eval_core_accuracy`):

```bash
# same metric across runs — metric named once
python -m cdrseqgen.visualization.linegraph_plotter \
  --metric eval_core_accuracy \
  --run "H3 Scratch" "results/20260729_202635/h3_OAS_scratch" \
  --run "H3 ESM-2"   "results/20260729_202635/h3_OAS_esm2_t12_35M_UR50D" \
  --title "H3 Core Recovery: Scratch vs ESM-2"

# breakdown of ONE run — per-run metric, several columns on one chart
python -m cdrseqgen.visualization.linegraph_plotter \
  --run whole  "results/runs/latest/h3_OAS_scratch" eval_accuracy \
  --run core   "results/runs/latest/h3_OAS_scratch" eval_core_accuracy \
  --run anchor "results/runs/latest/h3_OAS_scratch" eval_anchor_accuracy \
  --ylabel Recovery --title "H3 whole vs core vs anchor"
```

A missing CSV or column is reported and skipped rather than aborting the chart, so one bad path in a
multi-run comparison doesn't lose the other lines. Charts default to `results/figures/training/`.

<!-- > **⚠️ Never mix datasets on one chart.** OAS (`_OAS`) and SAbDab2 (`_SD2`) runs are evaluated on *different test sets*, so their accuracies are not comparable on shared axes — plot one graph per dataset (e.g. `cdr3_oas` with the 4 OAS runs, `cdr3_sd2` with the 4 SAbDab2 runs). The `results/comparisons/*_oas_*.png` / `*_sd2_*.png` pairs follow this convention. -->

## Step 4: Qualitative Sampling on Curated Frameworks (`generate.py`)
Give it a framework with its CDR loop cut out (`pt1` / gap / `pt2`) and it fills the gap: samples a
library of candidate loops (default **iterative**) and prints them, each with recovery-vs-native (when
you provide the native) and pseudo-perplexity, plus per-framework diversity/novelty. It also reports
which CDR the checkpoint was trained for (read from `run_config.json`). Samples + metrics are written to
`<model_dir>/generation_results.json`.

> **This is a showcase, not an evaluation.** `showcase/*.json` holds only a handful of curated
> frameworks (mostly without natives). For numbers you can report, use **Step 5**, which draws real
> held-out frameworks straight from the eval split.

> **⚠️ Match the checkpoint's context, or the model silently underperforms.** A checkpoint trained with
> partner-chain or antigen context does not *fail* on a bare framework — the input is merely
> out-of-distribution, so it quietly produces bad loops. Measured on `h1_OAS_LC_scratch`, 60 held-out
> frameworks, greedy: **0.9858** recovery with the light chain supplied as trained, **0.2599** without.
> In the latest batch **24 of 36 checkpoints** need context.
>
> Pass it with `--partner_chain` / `--antigen` (or per-case `partner_chain` / `antigen` keys in a
> `--test_file`), and set `--max_len` to the training config's value (512 for the SAbDab2 antigen runs).
> `generate.py` reads `run_config.json`, prints what the checkpoint expects, and **refuses** a context
> mismatch in either direction unless `--force` — supplying context a model never saw is as damaging
> as withholding context it needs (0.308 → **0.000** on `h3_OAS_scratch` with a partner chain
> prepended), because for an absolute-position backbone it shifts every learned position. Injection
> goes through the same `inject_context` / `assemble_masked_input` that build the training examples,
> so which side the antigen lands on — and which side gets truncated — follows the backbone's
> `position_type` automatically instead of being hand-encoded in the test file.
>
> The **loop** is a weaker signal and is treated accordingly. Nothing in the tensor names a loop or a
> chain, so which CDR you are designing is fixed by which checkpoint you load; `--target_loop` only
> *declares* your intent and cannot change the computation, so a mismatch **warns** rather than
> refuses — the input is still well-formed, just answered by weights fitted for another loop, which
> is a legitimate thing to probe. Because that check is opt-in, one more runs unconditionally: the
> length of `pt1` says where along the chain the gap sits, and across all 50,180 frameworks in both
> processed datasets the three CDR numbers separate cleanly (CDR1 8–26, CDR2 33–57, CDR3 78–106).
> That catches an h3 checkpoint fed an H1 framework with nothing declared, which was previously
> silent. Only the **number** is inferable — H1 (23–25) and L1 (24–26) overlap outright — so a
> heavy/light mix-up still cannot be detected this way.

```bash
# design coherent loops for a batch of curated frameworks (default: iterative sampling)
# results/runs/latest is the newest batch — name the config under it (here: h3_OAS_scratch)
# --model_dir is the checkpoint folder in generate.py, evaluate.py and fit_mlm_length.py; the first two also accept --model_path
python generate.py --model_dir ./results/runs/latest/h3_OAS_scratch --test_file showcase/h3.json --use_native_length

# more candidates + nucleus sampling
python generate.py --model_dir ./results/runs/latest/h3_OAS_scratch --test_file showcase/h3.json \
    --use_native_length --decoding iterative --num_samples 50 --top_p 0.9

# a single framework straight from the CLI (equal bounds mean an exact length)
python generate.py --model_dir ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --min_length 17 --max_length 17

# constrain each sampled length to an inclusive range (one-sided bounds also work)
python generate.py --model_dir ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --min_length 11 --max_length 15 --num_samples 50

# infer length from the framework: omit target_length/min_length/max_length
python generate.py --model_dir ./results/runs/latest/h3_OAS_scratch --pt1 EVQL...YYCAR --pt2 WGQGT... \
    --num_samples 50
```

Length selection passes through one explicit policy layer. Equal bounds request an exact fixed
length; unequal or one-sided bounds truncate and renormalize the checkpoint's learned length
distribution; omitted bounds sample that distribution without consulting the native target. A
`target_truth` is scoring-only unless `--use_native_length` is passed explicitly, preventing silent
length leakage. `--target_length` remains as a deprecated fixed-length alias. Every result stores the
policy, bounds, constrained probabilities, selected lengths, and prompt truncation under
`length_policy`.

The sequence backbone is still an encoder-only MLM: one **gap sentinel** is classified into a length,
then expanded into that many residue masks before the existing decoder runs. New training runs fit
and save `length_config.json` plus `length_head.safetensors` beside the best sequence checkpoint by
default; the sequence weights remain frozen during this fit. For a checkpoint trained before these
artifacts existed, run:

```bash
python fit_mlm_length.py --model_dir results/path/to/old_run
```

This discovers the run's saved `run_config.json`, keeping the original processed dataset and split.
Use `--config path/to/config.json` only when the checkpoint has no saved run configuration.
Inspect the resulting `length_config.json` before generation: its held-out `eval_accuracy` and
`eval_length_mae` should improve on `eval_mode_baseline_accuracy` and
`eval_mode_baseline_length_mae`; otherwise the head is mostly reproducing the training-set mode.

> **First result (2026-08-17): the mechanism works, but this length head does not yet beat the naive
> baseline.** On the full 2,111-framework validation split, the tuned scratch checkpoint's head
> reached 9.24% exact-length accuracy and 3.64 MAE, versus 10.52% and 3.28 for always predicting the
> training mode (length 15). See Finding §10; this is an open modeling result, not a software failure.

`--decoding`: `independent` (per-position sampling, 1 forward pass) · `iterative` (confidence-ordered
Mask-Predict — conditions each residue on those already committed). **`--temperature 0` makes either
deterministic (greedy)** and returns a single loop. Warp with `--temperature` / `--top_k` / `--top_p`;
`--skip_perplexity` drops the compute-heavy scoring.

> **⚠️ The two decoders are currently equivalent — use `independent` for sweeps.** Measured 2026-07-28
> on `h3_OAS_esm2`, 150 stratified frameworks, 20 samples, `top_p 0.9`, seed 42, IgBert judge:
>
> | metric | iterative | independent | Δ |
> |---|---|---|---|
> | recovery | 0.4057 | 0.4072 | +0.0015 |
> | diversity | 0.5731 | 0.5703 | −0.0028 |
> | entropy (bits) | 1.8414 | 1.8242 | −0.0171 |
> | judge-ppl | 7.1434 | 7.2259 | +0.0825 |
>
> Every difference is within noise, while iterative costs **~15×** the forward passes (one per masked
> position; mean H3 length 15.1). Mask-Predict only helps when the loop joint differs from the product
> of per-position marginals — getting identical numbers says this masked-LM has learned near-independent
> marginals for H3. That is a limit of the **training objective**, not of the sampler, and it is direct
> evidence for the "marginal-based, not learned-joint" open problem below.

## Step 5: Generative Evaluation on the Held-Out Split (`evaluate.py`)
The reportable version of Step 4. Reconstructs the **exact training-time eval split** from the config
(same clustered leakage-safe holdout, same partner/antigen injection, a real native on *every*
framework — 2,111 held-out antibodies for OAS), samples `--num_samples` candidates each, and aggregates
the generative metrics — overall **and bucketed by H3 length**, since difficulty scales with it.

```bash
python evaluate.py \
    --config configs/mlm/heavy/h3_esm2.json \
    --model_dir results/20260710/20260710_184547_h3_OAS_esm2_t12_35M_UR50D \
    --num_examples 150 --stratify --num_samples 20 \
    --top_p 0.9 \
    --judge Exscientia/IgBert
```

`evaluate.py` checks the supplied config against the checkpoint's `run_config.json` before loading
weights. A target-loop, backbone/tokenizer, context, dataset, maximum-length, or masking-policy
mismatch stops the run because it would change the model's inputs or invalidate the interpretation.
`--force` is reserved for deliberate off-contract experiments and records every mismatch in the
output. Decoder selection is also policy-aware: `full_span` checkpoints use `independent`;
`mixed` checkpoints use their recorded decoder (or `iterative` when none was recorded). Passing
`--decoding iterative` explicitly remains possible, with an off-distribution warning on a
`full_span` checkpoint.

The command above is the original `fixed_native` track. Use the checkpoint length head for a bounded
or fully inferred sweep:

```bash
# inclusive range; either min or max may be omitted
python evaluate.py --config configs/mlm/heavy/h3_esm2.json --model_dir results/path/to/run \
    --length_mode bounded --min_length 11 --max_length 15 --num_samples 20

# no native mask count is supplied to generation
python evaluate.py --config configs/mlm/heavy/h3_esm2.json --model_dir results/path/to/run \
    --length_mode inferred --num_samples 20
```

Positional AAR/core/anchor recovery is calculated only for candidates whose generated length matches
the native loop. All valid candidates still contribute to length-match rate, length MAE,
alignment-aware edit similarity, diversity, validity, runtime, and judge perplexity.

> **⚠️ Self-perplexity is NOT comparable across models.** Pseudo-perplexity is scored under each model's
> *own* distribution, so it measures self-consistency — a model confident in its own outputs looks good
> regardless of quality. To compare models, pass `--judge <model_type>`: every candidate is then also
> scored by one **independent** reference LM (IgBert is a good pick — antibody-specific, and independent
> of both scratch and ESM-2). Diversity/entropy carry no such caveat; they're purely sequence-based.

`--num_examples K` subsamples the split (`-1` = all; default 150, the same budget as the training-time
`gen_num_examples`); `--stratify` balances the draw across
short/medium/long H3 buckets so rare long loops aren't washed out. Manual results land in a
parameter-addressed file under `<model_dir>/evaluations/`, so a temperature or seed sweep cannot
silently replace the training-time result. Repeating the exact same output path is refused unless
`--overwrite` is explicit. The JSON records checkpoint/config/data hashes, repository state,
compatibility checks, decoder settings, and the anchor definition.

### Generative evaluation runs automatically after every training run
Once all epochs finish, `train.py` fits the frozen-backbone length head, samples designs from the
finished best model, and writes `generation_eval.json` beside `epoch_metrics.csv`. Both the length
fit and fixed-native generative evaluation are **on by default** — no config change is needed.

It samples from **`trainer.model`** (with `load_best_model_at_end` that's the peak checkpoint, already
on the GPU, so nothing is reloaded from disk), runs last, and is exception-guarded: a generation
failure logs a warning and **cannot** damage a training run that already finished and saved.

Every candidate is also scored by an **independent judge** (`gen_judge`, default
`Exscientia/IgBert`), which is what makes perplexity comparable *across* runs — self-perplexity is
not. Comparability requires the same judge in every run, **not** the same process, so per-run
judging is valid. Tune or disable via config:

```json
"run_generative_eval": true,
"gen_num_examples": 150,
"gen_num_samples": 20,
"gen_decoding": "independent",
"gen_top_p": 0.9,
"gen_length_mode": "fixed_native",
"gen_min_length": null,
"gen_max_length": null,
"gen_length_temperature": 1.0
```

Set `run_generative_eval: false` (or `gen_judge: ""` to keep the metrics but skip the judge) to opt out;
trim `gen_num_examples` for cheap runs. Cost is ~80s of sampling + ~15s to load the judge per run.

Both evaluation paths share the MLM evaluation implementation
([`cdrseqgen/methods/mlm/evaluation.py`](../cdrseqgen/methods/mlm/evaluation.py)); they differ only in
where the model comes from. **Both length tracks also share one loop**: `run_eval_loop` owns example
scoring, judging and framework-record assembly, while a *strategy* turns an eval index into
candidates. The method wrapper selects frameworks and handles reporting/output.
Both strategies live in [`mlm/length.py`](../cdrseqgen/methods/mlm/length.py).
`FixedNativeStrategy` reads the dataset's own
masked view (so the mask count is the native one and the context is exactly what training used);
`VariableLengthStrategy`
rebuilds the framework and lets the checkpoint's length head decide. Before this, the two tracks were
separate 130-line functions that had already drifted apart in four ways (progress bar, write guard,
summary statistic, empty-mask guard) — differences that make two tracks quietly incomparable.

Length policy, checkpoint-specific classification, prompt construction and span sampling live
in [`mlm/length.py`](../cdrseqgen/methods/mlm/length.py); independent/iterative decoder dispatch
lives in [`mlm/sampling.py`](../cdrseqgen/methods/mlm/sampling.py). The causal method lives
beside it in `methods/ar/`; future methods can do the same without inheriting MLM mask-token,
pseudo-perplexity, or length-expansion assumptions. The MLM primitives are shared by `generate.py`,
both held-out evaluation paths, and the CDRSeqGen benchmark adapter. Case generation,
variable-length evaluation and the benchmark use `length.sample_length_conditioned_spans`;
fixed-native evaluation directly decodes the dataset's existing masks. Both reach the
same `sampling.sample_masked_span` decoder dispatcher.
`evaluate.py` stays the tool of choice for
scoring checkpoints you already trained, or re-sweeping temperature / top-p / judge without
retraining.

## Matched External Benchmark

The frozen-manifest CDRSeqGen–IgLM validation and final-test workflow is documented in
[`benchmark/README.md`](../benchmark/README.md). Generation remains model-specific in the two adapters;
[`benchmark/scorer.py`](../benchmark/scorer.py) validates the matched contract and applies one shared
metric implementation. Validation-smoke results are development checks, not final benchmark claims.

Current length-free status (2026-08-17): the tuned scratch adapter completed all 25 validation
frameworks × 20 candidates with zero failures and a 6.0% sampled native-length match rate. IgLM's
matched natural-EOS output, shared scoring, and the comparison plot are still pending; therefore no
length-free CDRSeqGen-versus-IgLM claim exists yet.

---

## Preprocessing quick reference

Moved from the README on 26 October 2026. These source-specific commands supplement
Step 1 above; existing processed datasets do not need regeneration for this cleanup.

## 🧹 Data Preprocessing Quick Start

Run these commands from the `CDRSeqGen/` repository root after activating `cdrprj`. The
downloaders are optional when the raw files are already present; they resolve paths from the
repository root, so they can also be called from another working directory.

### OAS paired repertoires

Download and extract the eight current paired-repertoire CSVs:

```bash
bash cdrseqgen/datasets/preprocess/download_oas_paired.sh
```

Build one merged corpus and assign the split once across all donors. The default OAS pipeline
keeps productive paired chains, removes exact duplicate heavy/light pairs, clusters heavy CDR3s,
and writes both the processed JSON and its provenance sidecar:

```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
  --input_file data/raw/oas/*_Paired_All.csv \
  --output_file data/processed/processed_oas_multi.json \
  --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
```

For a quick input or environment check, apply `--limit` after source filtering:

```bash
python -m cdrseqgen.datasets.preprocess.oas_paired \
  --input_file data/raw/oas/1287169_1_Paired_All.csv \
  --output_file data/processed/processed_oas_smoke.json \
  --limit 100 --split none
```

Use `--split cluster` for model development. `--split none` is only suitable for inspection or
legacy reproduction because training must then create a random split at load time.

### SAbDab2 antibody-antigen pairs

Download the upstream split archive and extract it under `data/raw/sabdab2/`:

```bash
bash cdrseqgen/datasets/preprocess/download_sabdab2.sh
```

Process the paired structures with resolved protein or peptide antigens:

```bash
python -m cdrseqgen.datasets.preprocess.sabdab2 \
  --input_file data/raw/sabdab2/splits_final/abag_split.csv \
  --output_file data/processed/processed_sabdab2_data.json
```

The SAbDab2 processor preserves the upstream `train`/`test` split and does not create validation
records unless requested. To carve validation records from `train` without touching `test`, use
`--validation_size 0.15 --seed 42`; whole antigen/CDR-H3 groups are moved together. Add
`--allow_missing_antigen` only when intentionally building an antibody-only corpus.

Every command above also runs as `python -m cdrseqgen.datasets.preprocess --source
oas_paired|sabdab2 [flags]` -- the same pipeline, reached through `preprocess/__main__.py`.
Both processors accept `--help` for all source-specific options and write a `*.metadata.json`
sidecar containing the input/filter/split settings and a SHA-256 hash of the processed JSON. The
full funnel, split caveats, multi-donor guidance, and output schema are documented in
[WORKFLOW.md, Step 1](WORKFLOW.md#step-1-preprocessing-data).

---

## Historical S2S experiment notes

The following September preparation notes were moved from the README on 26 October 2026.
Preserve them as protocol and diagnostic history. Several commands describe batches
that later completed; use [RESULTS.md](RESULTS.md) and [PROJECT_STATUS.md](PROJECT_STATUS.md)
for current completion status before planning further work.

### S2S pretrained encoders: results and next runs

From the repository root, the combined suite is equivalent to listing all three model suites:

```bash
DRY_RUN=1 ./scripts/run.sh 0 s2s_plm   # inspect the plan; no training or result files
./scripts/run.sh 0 s2s_plm            # same as: 0 s2s_esm2 s2s_igbert s2s_protbert
# With multiple available GPUs:
# ./scripts/run.sh 0,1,2 s2s_plm
```

This schedules **six H3/OAS4 validation runs**: ESM-2, IgBert, and ProtBert, each with and without
partner-chain context. One GPU runs them sequentially; multiple GPUs run separate trials.
Defaults fine-tune the pretrained encoder and train a new 128-wide, two-layer residue decoder:
LR 3e-5, warmup 0.06, weight decay 0.01, effective batch 32, and up to ten epochs with early stopping.
These are initial settings, not tuned optima. Each run performs generative evaluation afterward
(150 frameworks × 20 candidates, top_p 0.9, seed 42, IgBert judge).

`FREEZE_ENCODER=true` keeps the encoder fixed; `ENCODER_INIT=random` gives a matching architecture
with random encoder weights. Both controls work in suite and sweep mode. The default matrix is
complete; tune and repeat promising settings across seeds before claiming a pretraining benefit.
See [the workflow](WORKFLOW.md#step-2-training--batch-experimentation) for control and sweep commands.

**Completed:** the [first full S2S PLM matrix](../results/runs/20260913_151152_463868_s2s_plm/batch_tables.md)
finished all six runs, including full ProtBert training and evaluation. All runs reached ten epochs;
each produced 3,000/3,000 nonempty, complete candidates with full judge coverage (18,000 total).
These are seed-42 **validation** results, not held-out test results:

| Encoder | Partner context | Judge perplexity ↓ | Exact length (%) ↑ | Length MAE ↓ | Generated core recovery (%) ↑ | Recovery frameworks / 150 |
|---|---|---:|---:|---:|---:|---:|
| ESM-2 | No | 3.441 | 8.93 | 3.832 | 21.95 | 117 |
| ESM-2 | Yes | 3.609 | 9.07 | 3.708 | 19.99 | 111 |
| IgBert | No | 3.499 | 8.10 | 3.915 | 20.84 | 113 |
| IgBert | Yes | 4.132 | 6.70 | 4.364 | 13.33 | 106 |
| ProtBert | No | 3.471 | 7.77 | 4.010 | 19.87 | 111 |
| ProtBert | Yes | 3.623 | 7.20 | 4.081 | 20.16 | 103 |

Generated core recovery is conditional on length-matched candidates; the contributing framework
populations differ. The [earlier scratch comparison](../results/runs/20260908_173143_compare/batch_tables.md)
scored **3.215 / 3.250** judge perplexity without / with partner context. No PLM setting improves
that metric here, but capacities, optimizer settings, and epoch budgets differ, so this does not
isolate the effect of pretraining. Five PLM runs achieved their best teacher-forced core recovery at
epoch 10; ESM-2 + partner peaked at epoch 7. A ten-epoch screen does not establish convergence.

**Stability and comparative performance are separate questions.** The collapse described below
is a stability failure in one of six runs. All six PLM settings also score worse than the earlier
scratch reference on judge perplexity; the collapse alone does not explain that broader result.
These measurements describe the tested settings, but do not isolate the effect of pretraining:
PLMs had a ten-epoch budget against scratch's twenty, five PLM runs reached their best
teacher-forced core recovery at the final epoch, and tuning effort and model capacities differ.
ESM-2 t12, the smallest encoder tested, has the lowest PLM judge perplexity. Budget and optimizer
settings are possible explanations for this ordering; architecture and pretrained representations
remain possible explanations too. A comparison with matched data/evaluation protocols and comparable
tuning budgets would assess the complete systems more fairly. Pretrained-versus-random initialization
of the same encoder architecture is the more direct control for the contribution of pretraining.

**September 17 status:** all 12 context-dropout trials and all four ESM-2 LR trials are complete. The [September 15–17 review](../results/misc/reviews/0915-0917-review.md) verifies their saved models, matched generation populations and scoring coverage. Dropout improves robustness to a missing partner at a small context-present edit cost (§14); `1e-4` improves ESM-2 teacher-forced recovery and judge perplexity without a consistent edit-similarity gain over `3e-5` (§15). Generation from the three earlier IgBert diagnostic checkpoints remains pending.

**Next steps:**

1. **IgBert + partner: representation collapse confirmed** by the
   [September 14 checkpoint diagnostics](../results/misc/diagnostics/20260914_igbert/REPORT.md).
   On 256 validation examples, swapping the framework or partner changes no predicted tokens;
   final encoder representations are nearly constant. The no-partner checkpoint and original
   pretrained encoder retain input variation. Training/validation inputs fit within 234 tokens
   (cap 384), ruling out truncation for this corpus. The initiating event remains unresolved;
   the epoch-2 loss/anchor deterioration is the leading interval to investigate. Next compare the
   completed original-LR, lower-LR and frozen-encoder diagnostic checkpoints generatively.
   All three retained conditioning through epoch 3; replication is still needed to estimate
   collapse incidence. The separate AR/S2S context-dropout screen is now complete (§14).
2. Add the **matched no-partner training controls** before varying their optimizer. The
   partner-present ESM-2 LR screen is complete (§15). The older command below is an optional
   no-partner LR follow-up, not the completed batch: three LRs × two seeds, up to 20 epochs,
   with generative evaluation enabled. Checkpoint selection remains teacher-forced core recovery;
   sweep ranking uses judge perplexity. Inspect recovery, length, and coverage alongside that ranking.
3. Compare promising settings with **random initialization of the same encoder architecture** and
   **frozen pretrained encoders**, using comparable tuning budgets, then replicate finalists across
   additional seeds. Keep the test split untouched during selection.
4. **Compare tuned scratch S2S and PLM systems under a matched protocol.** Use the same corpus,
   split, evaluation examples, decoding settings, and scorer, with comparable tuning effort and
   explicit training budgets. Matching epoch budgets controls data exposure; it does not match
   compute across model sizes. Running the arms in one batch simplifies bookkeeping, but consistent
   protocol and provenance also permit comparison across batches. This comparison measures the
   complete systems; step 3's same-architecture initialization control addresses pretraining.

The immediate diagnostic [experiment spec](../configs/experiments/20260914_igbert_diagnostic.json) reuses the IgBert + partner config for three arms: trainable at
LR 3e-5, trainable at 1e-5, and frozen pretrained encoder at 3e-5. Seed 42, context dropout 0,
and the original ten-epoch scheduler horizon are retained; a callback stops training after epoch 3.
All three epoch checkpoints are retained, including optimizer state. Generation is disabled for
this diagnostic run. `conditioning_probe.jsonl` records projected encoder variation and sensitivity
to swapping whole encoder memories between 16 fixed validation examples, before training, every
250 optimizer steps, and at epoch ends. These probes do not select checkpoints or alter training RNG.
The frozen arm still trains its projection and decoder. Do not set `FREEZE_ENCODER` or `ENCODER_INIT`
globally in spec mode; put seed and encoder controls in the file. The recipe is an experiment
record, separate from the permanent suite catalog. Generic diagnostics remain available to future runs.

**How to read it.** The trainable-3e-5 arm is intended to reproduce the original first three epochs
with the same seed, data seed, context dropout and scheduler horizon. If it does not collapse,
report that the failure was not reproduced under this rerun; a fixed seed alone does not establish
identical numerical execution. The lower-LR and frozen arms still provide stability and conditioning
measurements, but a non-collapsing reproduction control would prevent claiming that those changes
averted the original failure on the basis of this comparison alone.

The earlier checkpoint diagnostic used 256 validation examples and swapped raw framework flanks;
it observed ~28.67% changed argmax tokens (mean absolute logit change ~0.577) for the no-partner
checkpoint versus 0% (~6.6e-7) for the collapsed checkpoint. The running callback uses 16 fixed
validation examples and swaps entire encoder memories, including partner context. Those historical
values illustrate the failure; they do not calibrate a pass threshold for the callback. Interpret
projected encoder variation and swap sensitivity together over training. No numerical pass/fail
criterion was specified before this batch started. Any threshold chosen from these traces is
exploratory and should be fixed before a subsequent confirmation run.

With 122,181 training records, microbatch 4 and accumulation 8, this run has **3,819 optimizer
steps per epoch**, including the final partial accumulation window. Three complete epochs produce
45 periodic probes at 250-step intervals, plus one initial and three epoch-end probes: **49 records
per arm**. The original training-loss transition was logged around epochs 1.83–2.09; validation
anchor accuracy was measured at epoch boundaries, falling from 88.19% at epoch 1 to 81.35% at
epoch 2. The probes can narrow the transition interval but do not identify its exact update.

Each retained checkpoint in the earlier fully trainable IgBert run occupied approximately
**4.69 GiB**, including optimizer state. The frozen arm still saves the full encoder weights,
but its optimizer state covers only the trainable projection and decoder, so its checkpoints
should be smaller. Budget the two trainable arms and the frozen arm separately, plus final
model exports and logs; multiplying the fully trainable checkpoint size by all nine checkpoints
overestimates the expected total.

```bash
DRY_RUN=1 ./scripts/run.sh 0 spec configs/experiments/20260914_igbert_diagnostic.json
./scripts/run.sh 0 spec configs/experiments/20260914_igbert_diagnostic.json
# Or run the three arms concurrently on three available GPUs:
# ./scripts/run.sh 0,1,2 spec configs/experiments/20260914_igbert_diagnostic.json
```

After reviewing stability, the ESM-2 tuning command is:

```bash
# Inspect the six-trial plan first; remove DRY_RUN=1 to launch. Choose available GPU IDs.
DRY_RUN=1 LRS="1e-5 3e-5 1e-4" SEEDS="42 43" EPOCH_LIST="20" \
GEN_EVAL=true RANK_METRIC=judge_ppl \
./scripts/sweep.sh 0 lr configs/s2s/h3_s2s_esm2.json
```

The [scratch S2S LR screen](../results/runs/20260907_204721_sweep_lr/summary.tsv) and the earlier
six-run MLM/AR/S2S comparison are also complete. The 414-test CPU suite passed on September 10;
that historical check and the completed GPU matrix establish execution coverage, not biological
quality or a pretraining advantage.
