# IgLM adaptation test runs — 2 November 2026

**Replay, zero-shot, IgLM-S fine-tuning and controls complete; large-model fine-tuning pending.** The full
GPU replay matched 150/150 examples and 3,000/3,000 attempts, with matching weights (210.9 s). Both
zero-shot trials, the fresh four-trial IgLM-S batch and the four-trial control batch completed with full
validity and judge coverage. See the [November 3 audit](../../results/reviews/20261103-iglm-adaptation.md),
the [November 5 controls audit](../../results/reviews/20261105-iglm-controls.md) and the
[replay report](../../results/controls/iglm_replay_full_20261102.json).
The declared validation-core rule selects lr 3e-5, epoch 11 (core accuracy 0.4072). The random-init IgLM-S
controls reach 0.4007–0.4063, all at the 20-epoch cap, and the scratch AR of IgLM-S's size 0.4037; generated
edit similarity is 0.3887–0.3903 for the controls and 0.3897 for the selected pretrained run. This is a
single-seed pilot with partner-present IgBert judging. `iglm_l_finetune` remains pending.

The fresh batch `20261103_165217_376850_iglm_s_finetune` supersedes the earlier stopped
batch for completion. No replay, zero-shot or fine-tuning completion rerun is needed.

## Scientific purpose

Test that the AR `iglm` backbone ([MODELS.md](../../docs/MODELS.md#pretrained-ar-backbone-iglm)) works, in four
steps of increasing cost:

1. **Protocol parity.** Compare the in-process sampler with the audited external IgLM row under matching settings.
2. **Trains.** Fine-tuning on OAS4 H3 produces finite losses and correctly accounted generation attempts.
3. **What pretraining contributes.** Pretrained against random-init weights of the same architecture and layout,
   with a dimension-matched scratch AR as an additional reference.
4. **Usable in the benchmark.** A fine-tuned IgLM arm runs through the benchmark adapter and judge on GPU.

Already shown on the CPU and not repeated here ([CHANGELOG](../../docs/CHANGELOG.md),
[archived verification evidence](../../results/reviews/archives/20261105-temp-cleanup.md)):
perplexity matches the upstream package to 3e-6 and greedy decoding is
identical in 240/240 cases; draws seeded identically equal the upstream package's own model in 90/90, across
transformers 4.57 and 5.17; a tiny fine-tune runs through train, evaluate, generate, the benchmark adapter and the scorer.
Those prior parity diagnostics included some test-split records. No final comparative held-out evaluation has
run, but the split has not been wholly unseen during implementation. Further software checks use synthetic
or validation examples; these parity outputs are not tuning evidence or new benchmark rows.

## Recipes and fixed protocol

| Recipe | Trials | What it answers |
|---|---:|---|
| [`iglm_zeroshot`](20261102_iglm_zeroshot.json) | 2 | IgLM-S and IgLM with `do_train: false`: validation loss and the full 150×20 generation evaluation, untrained |
| [`iglm_s_finetune`](20261102_iglm_s_finetune.json) | 4 | IgLM-S fine-tuned at lr 1e-5, 3e-5, 1e-4, 3e-4 |
| [`iglm_controls`](20261102_iglm_controls.json) | 4 | Random-init IgLM-S at lr 3e-5, 1e-4, 3e-4, and a scratch AR of IgLM-S's size (192/3/6/768) |
| [`iglm_l_finetune`](20261102_iglm_l_finetune.json) | 2 | IgLM (12.9M parameters) fine-tuned at lr 1e-5 and 3e-5 |

Shared base: [bases/h3_ar_iglm.json](bases/h3_ar_iglm.json). Every trial equals the completed
[`ar_scratch_no_partner_s42`](20261006_no_partner_controls.json) control except where the table says otherwise;
`test_iglm_experiments` pins this.

- Corpus OAS4 `data/processed/processed_oas_multi.json`, validation selection, H3, no partner or antigen, dropout 0, `max_len` 384
  (IgLM needs 101 to 512; H3 flanks are 107 residues, so nothing is truncated).
- Training and data seed 42 only: this is a feasibility test, so every difference is exploratory.
- Selection and generation as the completed controls: up to 20 epochs, patience 3, best validation core accuracy; 150 validation
  frameworks × 20 draws, generation seed 42, top-p 0.9, top-k 0, temperature 1, decode cap 97, IgBert with partner context (Table 8's policy).
- **Optimizer differs from the scratch control** and follows the ESM-2 pretrained controls: warmup 6%, weight decay 0.01 (scratch: none),
  batch 32, no accumulation. Learning rates are a grid, not a tuned optimum. The scratch control keeps its own lr 1e-4.
- Released checkpoints are read from `~/miniconda3/envs/iglm/lib/python3.11/site-packages/iglm/trained_models/`. If that environment
  moves, change `model_type` in the base and the two spec files; trial identities change with it.
- The random-init control is `results/controls/iglm_s_random_init`, made by `scripts/make_iglm_randinit.py` (seed 0, weights hash in its
  `RANDOM_INIT.json`). It must exist before `iglm_controls` starts; the runner does not create it.
- IgLM saw OAS in pretraining. Keep these rows in their own category in any table; they do not isolate architecture.

## Commands (user-run, GPU 0 only; GPU 1 is shared)

Preflight, no GPU. `plan` checks the corpus but not the backbone. The unit suite skips unavailable local
IgLM sources, so passing it alone does not establish that all experiment inputs exist. Run the explicit
source/tokenizer checks below as well (they read metadata, not model weights):

```bash
conda activate cdrprj
test -d results/controls/iglm_s_random_init || python scripts/make_iglm_randinit.py
for n in iglm_zeroshot iglm_s_finetune iglm_controls iglm_l_finetune; do ./scripts/run.sh plan $n --gpus 0; done
python -m unittest unit_tests.test_arguments_contracts.ShippedConfigTests unit_tests.test_iglm_experiments
python - <<'PY'
from types import SimpleNamespace
from cdrseqgen.experiments import spec_trials
from cdrseqgen.methods.ar.backbones import resolve
for name in ('iglm_zeroshot', 'iglm_s_finetune', 'iglm_controls', 'iglm_l_finetune'):
    trials, _ = spec_trials(f'configs/experiments/20261102_{name}.json', {})
    for trial in trials:
        config = trial['config']
        if config['model_type'] == 'scratch':
            continue
        backbone = resolve(config['model_type'])
        data = SimpleNamespace(**{'include_partner_chain': False, 'include_antigen': False, **config})
        backbone.validate_training_args(SimpleNamespace(**config), data)
        backbone.build_tokenizer(config['model_type'])
        print(name, trial['suffix'], 'source/tokenizer OK')
PY
```

The existing prepared control's missing tokenizer/layout metadata was repaired during the review;
its recorded weight hash is unchanged. The original config and repair record are retained beside it.
On another checkout, create the control if absent; the helper refuses replacement unless `--force`
is explicitly supplied and rejects overlapping source/output directories.

1. **Replay — completed.** The command is retained for reference; no repeat is needed. The stored
   batch was generated on a GPU, and CPU and CUDA random streams differ:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/iglm_replay.py \
       --batch results/benchmark/oas4_h3_length_free_20261007 --device cuda \
       --output results/controls/iglm_replay_20261102.json
   ```

   First add `--limit 5` for a short check, then replay all 150. The configured screening threshold is weights matching
   and at least 148 of 150 examples reproducing all 20 attempts. This threshold is not proof of equivalence:
   investigate every discrepancy in prompts, seeds, sampling settings, device and runtime. Floating-point differences
   are one possible cause of sustained divergence; the mismatch pattern alone cannot establish the cause.
2. **Remaining batch: the large model, if retained** (a new directory). `iglm_zeroshot`,
   `iglm_s_finetune` and `iglm_controls` are complete; they were launched the same way
   (`./scripts/run.sh run <recipe> --gpus 0`) and need no repeat:

   ```bash
   ./scripts/run.sh run iglm_l_finetune --gpus 0
   ./scripts/run.sh status results/runs/<printed-batch-directory>
   ```

   Each `run` prints its batch directory and writes the batch tables and plots at the end. Do not edit `cdrseqgen/` or `train.py` between
   `run` and `resume`: the plan pins the source.
3. **Benchmark smoke (after step 2).** `arms_recipe.py` refuses the zero-shot batch, which keeps no checkpoint.

   ```bash
   python benchmark/arms_recipe.py --name iglm_arms --batch results/runs/<s_finetune batch> \
       --output benchmark/recipes/generated/iglm_arms.json
   # keep only the arms you want (labels are the trial names), then:
   ./scripts/run.sh benchmark plan --recipe benchmark/recipes/generated/iglm_arms.json --stage smoke
   ./scripts/run.sh benchmark smoke --recipe benchmark/recipes/generated/iglm_arms.json --gpu 0
   ```

## Software checks and research diagnostics

| Question | Read | Pass (proposed) |
|---|---|---|
| Protocol parity? | replay report | weights match, no missing attempts, exact matches counted, discrepancies investigated |
| Zero-shot runs? | `iglm_zeroshot` generation evaluation | every attempt accounted for; nonempty candidates have finite scores; report validity and perplexity without imposing an expected score |
| Fine-tuning runs? | `epoch_metrics.csv` and generation outputs | finite losses, restored best checkpoint, complete attempt accounting; improvement and validity rate are measured outcomes |
| Pretraining helps? | same-architecture pretrained/random-init runs at the same lr | exploratory contrast under the declared protocol; either direction is a finding, not a software failure |
| Scratch reference? | dimension-matched scratch AR | dimensions match; vocabulary, positions, dropout, biases and weight tying also differ, so this does not isolate architecture |
| Benchmark arm works? | smoke batch | attempts and failures accounted for, valid candidates judge-scored, saved tokenizer/policy pinned, `prompt.conditioning` present |

Compare with the completed scratch rows in `results/runs/*no_partner_controls*` (`ar_scratch_no_partner_s42`). Choose the lr for the
benchmark by validation core accuracy. Do not use the test split for model or hyperparameter selection.

## Estimated cost (single GPU, sequential)

Runtime for these four recipes is unmeasured. Use the replay smoke check and first training epoch to estimate
the remaining time on the selected GPU; include generation and judging in the budget.

## CPU dry run

Before this note was written, all four recipes ran end to end through `scripts/experiments.py run` on tiny data, with a GPU id that does
not exist and `use_cpu`: plan, trials, artifact verification, batch tables and plots. The replay script was checked against a fresh CPU
run of the real external adapter (5 frameworks × 20 draws: 100/100 attempts reproduced) and fails, as it should, with the wrong weights.
Against the real stored batch on the CPU it reproduces 0 of 200 draws because the random streams differ, while its weights, seed and
prompt checks pass.
