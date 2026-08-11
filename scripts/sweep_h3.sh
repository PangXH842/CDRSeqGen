#!/usr/bin/env bash
#
# Validation-safe hyperparameter sweep for CDR-H3.
#
# Default screen: 4 learning rates x 2 warmups x 2 weight decays x 2 seeds = 32 trials.
# Every trial uses the same maximum epoch/scheduler horizon; early stopping observes ONLY the named
# validation split. The final test split is never selected for training, checkpointing, or scoring.
#
# Before the first run, create a three-way clustered dataset:
#   python -m cdrseqgen.data.preprocess.oas_paired \
#     --input_file dataset/data/raw/1287169_1_Paired_All.csv \
#     --output_file dataset/data/processed/processed_oas_tuning.json \
#     --split cluster --test_size 0.1 --validation_size 0.1 --seed 42
#
# Usage:
#   ./scripts/sweep_h3.sh <gpu> [base_config]
#
# Common overrides:
#   SEEDS="42 43 44" LRS="3e-4 1e-3" ./scripts/sweep_h3.sh 0
#   SWEEP_DIR=results/sweeps/h3_my_run ./scripts/sweep_h3.sh 0  # resume same directory
#   DATA_PATH=path/to/three_way.json MAX_EPOCHS=40 PATIENCE=5 ./scripts/sweep_h3.sh 0

set -uo pipefail

GPU=${1:-0}
BASE=${2:-configs/h3_scratch.json}
PYTHON=${PYTHON:-python}
DATA_PATH=${DATA_PATH:-dataset/data/processed/processed_oas_tuning.json}

LRS=${LRS:-"1e-4 3e-4 1e-3 3e-3"}
WARMUPS=${WARMUPS:-"0.0 0.06"}
WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0 0.01"}
SEEDS=${SEEDS:-"42 43"}
MAX_EPOCHS=${MAX_EPOCHS:-30}
PATIENCE=${PATIENCE:-5}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.0001}
SCHEDULER=${SCHEDULER:-linear}

[[ -f "$BASE" ]] || { echo "ERROR: no base config: $BASE"; exit 1; }
[[ -f "$DATA_PATH" ]] || {
  echo "ERROR: no three-way tuning dataset: $DATA_PATH"
  echo "Create it with the preprocessing command in this script's header."
  exit 1
}

# Refuse to tune on a two-way file or on the final test split.
"$PYTHON" - "$BASE" "$DATA_PATH" <<'PY'
import json
import sys
from collections import Counter

base_path, data_path = sys.argv[1:]
with open(base_path) as f:
    base = json.load(f)
with open(data_path) as f:
    data = json.load(f)

if base.get("target_region", "h3").lower() != "h3":
    raise SystemExit("ERROR: this sweep is deliberately H3-only; base config targets "
                     f"{base.get('target_region')!r}")
counts = Counter(row.get("split") for row in data)
required = {"train", "validation", "test"}
missing = required - set(counts)
if missing:
    raise SystemExit(
        f"ERROR: {data_path} has split counts {dict(counts)}; missing {sorted(missing)}. "
        "Do not tune on the historical two-way train/test file."
    )
print(f"Dataset gate passed: train={counts['train']}, validation={counts['validation']}, "
      f"test={counts['test']} (test will not be selected).")
PY
[[ $? -eq 0 ]] || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"

export CDRSEQGEN_SKIP_LATEST=1
STAMP=$(date +'%Y%m%d_%H%M%S')
OUT=${SWEEP_DIR:-"results/sweeps/h3_$STAMP"}
mkdir -p "$OUT"
export CDRSEQGEN_RESULTS_ROOT="$OUT"

MAP="$OUT/runs.tsv"
SUMMARY="$OUT/summary.tsv"
if [[ ! -f "$MAP" ]]; then
  printf 'lr\twarmup\tweight_decay\tseed\tmax_epochs\tscheduler\tstatus\trun_dir\n' > "$MAP"
fi

total=0
for lr in $LRS; do
  for wu in $WARMUPS; do
    for wd in $WEIGHT_DECAYS; do
      for seed in $SEEDS; do
        total=$((total + 1))
      done
    done
  done
done

echo "======================================================================"
echo "H3 validation sweep | GPU $GPU | base=$BASE | trials=$total"
echo "data=$DATA_PATH"
echo "LR=[$LRS] warmup=[$WARMUPS] weight_decay=[$WEIGHT_DECAYS] seeds=[$SEEDS]"
echo "max_epochs=$MAX_EPOCHS patience=$PATIENCE threshold=$STOP_THRESHOLD scheduler=$SCHEDULER"
echo "output=$OUT"
echo "======================================================================"

i=0
for lr in $LRS; do
  for wu in $WARMUPS; do
    for wd in $WEIGHT_DECAYS; do
      for seed in $SEEDS; do
        i=$((i + 1))

        if awk -F '\t' -v lr="$lr" -v wu="$wu" -v wd="$wd" -v seed="$seed" \
          'NR > 1 && $1 == lr && $2 == wu && $3 == wd && $4 == seed && $7 == "completed" {found=1} END {exit !found}' \
          "$MAP"; then
          echo "[$i/$total] SKIP completed: lr=$lr warmup=$wu wd=$wd seed=$seed"
          continue
        fi

        tag="lr${lr}_wu${wu}_wd${wd}_s${seed}"
        export CDRSEQGEN_RUN_SUFFIX="$tag"
        LOG="$OUT/${tag}.log"

        echo "[$i/$total] RUN  lr=$lr warmup=$wu wd=$wd seed=$seed"
        if "$PYTHON" train.py "$BASE" \
          --data_path "$DATA_PATH" \
          --eval_split validation \
          --learning_rate "$lr" \
          --warmup_ratio "$wu" \
          --weight_decay "$wd" \
          --num_train_epochs "$MAX_EPOCHS" \
          --lr_scheduler_type "$SCHEDULER" \
          --early_stopping_patience "$PATIENCE" \
          --early_stopping_threshold "$STOP_THRESHOLD" \
          --metric_for_best_model core_accuracy \
          --greater_is_better true \
          --load_best_model_at_end true \
          --save_total_limit 1 \
          --seed "$seed" \
          --data_seed "$seed" \
          --run_generative_eval false \
          > "$LOG" 2>&1; then
          RUNDIR=$(grep -oE "Workspace initialized at: .*" "$LOG" | tail -1 | sed 's/.*at: //')
          if [[ -n "$RUNDIR" && -f "$RUNDIR/epoch_metrics.csv" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\tcompleted\t%s\n' \
              "$lr" "$wu" "$wd" "$seed" "$MAX_EPOCHS" "$SCHEDULER" "$RUNDIR" >> "$MAP"
            echo "          OK: $RUNDIR"
          else
            printf '%s\t%s\t%s\t%s\t%s\t%s\tincomplete\t%s\n' \
              "$lr" "$wu" "$wd" "$seed" "$MAX_EPOCHS" "$SCHEDULER" "${RUNDIR:-}" >> "$MAP"
            echo "          INCOMPLETE: training exited successfully but metrics/run path is missing; see $LOG"
          fi
        else
          printf '%s\t%s\t%s\t%s\t%s\t%s\tfailed\t\n' \
            "$lr" "$wu" "$wd" "$seed" "$MAX_EPOCHS" "$SCHEDULER" >> "$MAP"
          echo "          FAILED: see $LOG"
        fi
      done
    done
  done
done

"$PYTHON" - "$MAP" "$SUMMARY" <<'PY'
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

map_path, summary_path = sys.argv[1:]
trials = []
with open(map_path) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["status"] != "completed":
            continue
        metrics_path = os.path.join(row["run_dir"], "epoch_metrics.csv")
        if not os.path.exists(metrics_path):
            continue
        with open(metrics_path) as mf:
            metrics = list(csv.DictReader(mf))
        if not metrics:
            continue
        best = max(metrics, key=lambda x: float(x["eval_core_accuracy"]))
        trials.append({
            **row,
            "best_core": float(best["eval_core_accuracy"]),
            "best_epoch": float(best["epoch"]),
            "last_epoch": float(metrics[-1]["epoch"]),
        })

groups = defaultdict(list)
for trial in trials:
    key = (trial["lr"], trial["warmup"], trial["weight_decay"])
    groups[key].append(trial)

ranked = []
for (lr, wu, wd), members in groups.items():
    values = [m["best_core"] for m in members]
    epochs = [m["best_epoch"] for m in members]
    ranked.append({
        "lr": lr,
        "warmup": wu,
        "weight_decay": wd,
        "n_seeds": len(values),
        "mean_core": statistics.mean(values),
        "sd_core": statistics.stdev(values) if len(values) > 1 else math.nan,
        "min_core": min(values),
        "max_core": max(values),
        "mean_best_epoch": statistics.mean(epochs),
    })
ranked.sort(key=lambda x: (-x["mean_core"], float(x["lr"])))

fields = ["lr", "warmup", "weight_decay", "n_seeds", "mean_core", "sd_core",
          "min_core", "max_core", "mean_best_epoch"]
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(ranked)

print("\nValidation ranking (mean best core recovery across seeds)")
print(" lr       warmup  wd      n   mean      sd        min       max       mean_best_ep")
for r in ranked:
    sd = "n/a" if math.isnan(r["sd_core"]) else f"{r['sd_core']:.6f}"
    print(f" {float(r['lr']):.0e}   {float(r['warmup']):>5.2f}   "
          f"{float(r['weight_decay']):>5.3f}  {r['n_seeds']:>2}  "
          f"{r['mean_core']:.6f}  {sd:>8}  {r['min_core']:.6f}  "
          f"{r['max_core']:.6f}  {r['mean_best_epoch']:.1f}")

if ranked:
    winner = ranked[0]
    print("\nScreening winner (still validation-only): "
          f"lr={winner['lr']}, warmup={winner['warmup']}, "
          f"weight_decay={winner['weight_decay']}, mean={winner['mean_core']:.6f}")
    if winner["n_seeds"] < 2:
        print("WARNING: winner has fewer than two completed seeds; do not freeze it yet.")
print(f"Summary written to {summary_path}")
PY

echo "Done. The test split was not used for training, checkpoint selection, or scoring."

