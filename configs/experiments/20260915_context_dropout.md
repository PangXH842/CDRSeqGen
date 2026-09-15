# H3 context-dropout validation screen — 2026-09-15

This is a prospective exploratory screen prepared before these training runs. It is not the final paper test protocol. No GPU experiments were launched during preparation; the user will launch them. The earlier IgBert diagnostic batch is complete, while generation from its three saved checkpoints is pending.

## Question and controls

Does training with partner-context dropout improve robustness when the partner chain is absent, and what does it cost when the partner is present?

The [spec](20260915_context_dropout.json) schedules 12 trials: scratch AR, scratch S2S, and ESM-2 S2S × dropout 0/0.5 × training seeds 42/43. Both dropout conditions are configured to accept partner context. Dropout 0 is the partner-trained specialist, not a separately trained no-partner model. The drop assignment is deterministic per example, context kind and seed; it is not resampled every epoch.

Within each model/seed pair, only `context_dropout` changes. All trials use OAS4 H3, the existing validation split, max length 384, up to 20 epochs, patience 3, and checkpoint selection by teacher-forced core accuracy. Validation always supplies partner context and disables context dropout. This deliberately selects for the context-present regime; robustness without the partner is a secondary evaluation, not a checkpoint-selection signal. Seed controls model/training randomness and data order. ESM-2 uses its existing LR 3e-5, warmup 0.06, decay 0.01, effective batch 32; scratch models retain LR 1e-4 and zero warmup/decay at batch 32. Comparisons are paired within each model, not a capacity- or optimizer-matched architecture ranking.

## Fixed generation protocol

- Same 150 validation frameworks, selected with seed 42 without length stratification; 20 candidates per framework per regime, temperature 1, top-p 0.9, top-k 0, natural EOS and the existing 97-step cap. No native loop length enters generation.
- Each checkpoint is evaluated with and without partner context. Evaluation dropout is always 0. The same selection seed fixes example identities across models, training seeds and regimes; decoding randomness is reproducible within an evaluation but different model outputs can consume it differently.
- The IgBert judge receives the same full partner context in both regimes. Its scores are supporting measures of conditional plausibility, not a sole quality endpoint. This is a shared pretrained reference; distribution overlap and model-family affinity remain limitations.
- Primary screen comparison: change in `edit_similarity_mean` from dropout 0 to 0.5, reported separately for each evaluation regime and training seed. This metric averages per-framework native edit similarity over completed, nonempty candidates; interpret it with validity and contributing-framework counts. If coverage differs, report a common-framework comparison as well and do not attribute the aggregate change solely to quality.
- Supporting metrics: validity across all attempts; length-match rate and capped length MAE; core/anchor recovery conditional on matching length and its contributing populations; diversity, uniqueness/novelty and judge perplexity. Native similarity measures reconstruction, not binding function.
- Report the context-present/context-absent tradeoff and paired per-framework differences. Two seeds provide screening evidence, not a stable estimate of training variability. Do not pick a winner solely from a pooled mean or claim equivalence from a nonsignificant result.
- No test-set inspection or tuning. Select follow-up configurations on validation; add seeds, finalize endpoints and audit split/pretraining overlap before the paper's held-out confirmation.

Outputs per checkpoint: `generation_eval.json` (with partner) and `generation_eval_without_partner.json` (without partner). Both record evaluation/training context and judge context. `batch_tables.md` includes a partner-context ablation table; neither regime overwrites the other. The executor requires both outputs before marking a trial complete.

## Commands

From the repository root, choose available GPU IDs. The launcher activates the project environment. A single GPU runs trials sequentially; multiple IDs shard the trials.

```bash
DRY_RUN=1 ./scripts/run.sh 0 spec configs/experiments/20260915_context_dropout.json
./scripts/run.sh 0 spec configs/experiments/20260915_context_dropout.json
```

Replace `0` with the GPU(s) you intend to use. Record the final reviewed code revision before launching and avoid changing runtime code while queued trials are running. Use `EXPERIMENT_DIR=<batch>` with an unchanged resolved plan for runner resume; interrupted training trials are restarted, not resumed from an optimizer checkpoint.

## Evaluate the completed IgBert diagnostic checkpoints first

These evaluations use the same 150 × 20 generation protocol, with partner context present for both generator and judge. Their three-epoch, single-seed results are a diagnostic comparison, not a final frozen-versus-fine-tuned claim. This command loads each run's exact resolved input config from its batch; it does not retrain or replace checkpoint weights. Results go into each run's `evaluations/` directory; identical existing output is protected against overwrite.

Activate `cdrprj` and set `CUDA_VISIBLE_DEVICES` to the GPU you choose before running:

```bash
conda activate cdrprj
export CUDA_VISIBLE_DEVICES=0  # replace with your chosen GPU
python - <<'PY'
import csv
import subprocess
import sys
from pathlib import Path

batch = Path('results/runs/20260914_140305_226060_igbert_diagnostic')
with (batch / 'runs.tsv').open() as handle:
    rows = list(csv.DictReader(handle, delimiter='\t'))
for row in rows:
    if row['status'] != 'completed':
        raise RuntimeError(f"Unfinished diagnostic trial: {row['trial_id']}")
    subprocess.run([
        sys.executable, 'evaluate.py',
        '--config', str(batch / '.configs' / (row['trial_id'] + '.json')),
        '--model_dir', row['run_dir'],
        '--num_examples', '150', '--num_samples', '20',
        '--temperature', '1', '--top_k', '0', '--top_p', '0.9', '--seed', '42',
        '--judge', 'Exscientia/IgBert',
        '--partner_context', 'with', '--judge_partner_context', 'with',
    ], check=True)
PY
```

For a later manual ablation of a single checkpoint, keep its original config and set `--partner_context without --judge_partner_context with`. This explicitly changes generator context while preserving the original checkpoint compatibility check and the judge context. The output filename includes both context choices.
