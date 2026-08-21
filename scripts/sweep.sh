#!/usr/bin/env bash
#
# Unified hyperparameter sweep for CDRSeqGen. Replaces sweep_h3.sh and sweep_scratch.sh.
#
# Usage:  ./scripts/sweep.sh <gpu> [preset] [base_config]      (default preset: h3)
#
# Presets (one at a time; every dimension is env-overridable, see below):
#   h3       validation-safe screen: 4 LR x 2 warmup x 2 weight-decay x 2 seeds   (32 trials)
#            Fixed 30-epoch scheduler horizon — the budget is deliberately NOT swept, because
#            changing num_train_epochs also reshapes the linear schedule, which would confound
#            "more epochs" with "different schedule".
#   pilot    the 2026-07 question-answering grid: 3 LR x 2 epoch budgets          (6 trials)
#            Kept because it answers a different question — is the budget itself enough? — at
#            the cost of confounding budget with schedule. Use h3 to select hyperparameters.
#   arch     architecture screen: 3 widths x 2 depths, FFN tracks width           (6 trials)
#            Only valid since 2026-07-27, when num_attention_heads / intermediate_size became
#            ModelArguments; before that the FFN ratio silently changed with hidden_size.
#   quick    2 trials, 2 epochs — for checking this script runs, not for results.
#
# Every preset selects on the named validation split and never touches test. The dataset gate
# below refuses a two-way train/test file outright.
#
# Overrides (any preset):
#   LRS="3e-4 1e-3" SEEDS="42 43 44" ./scripts/sweep.sh 0 h3
#   MAX_EPOCHS=40 PATIENCE=5 ./scripts/sweep.sh 0 h3
#   SWEEP_DIR=results/runs/my_sweep ./scripts/sweep.sh 0 h3     # resume into the same directory
#   DATA_PATH=path/to/three_way.json ./scripts/sweep.sh 0 h3
#
# Before the first run, create a three-way clustered dataset:
#   python -m cdrseqgen.data.preprocess.oas_paired \
#     --input_file dataset/data/raw/1287169_1_Paired_All.csv \
#     --output_file dataset/data/processed/processed_oas_tuning.json \
#     --split cluster --test_size 0.1 --validation_size 0.1 --seed 42

set -uo pipefail

GPU=${1:-0}
PRESET=${2:-h3}
BASE=${3:-configs/h3_scratch.json}
PYTHON=${PYTHON:-python}
DATA_PATH=${DATA_PATH:-dataset/data/processed/processed_oas_tuning.json}

# --- Preset grids. Env vars win, so a preset only supplies defaults. --------------------------
case "$PRESET" in
  h3)
    LRS=${LRS:-"1e-4 3e-4 1e-3 3e-3"}; WARMUPS=${WARMUPS:-"0.0 0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0 0.01"}; SEEDS=${SEEDS:-"42 43"}
    EPOCH_LIST=${EPOCH_LIST:-"30"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  pilot)
    LRS=${LRS:-"1e-4 3e-4 1e-3"}; WARMUPS=${WARMUPS:-"0.0"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"10 20"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  arch)
    LRS=${LRS:-"3e-4"}; WARMUPS=${WARMUPS:-"0.06"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.01"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"30"}
    HIDDEN_SIZES=${HIDDEN_SIZES:-"128 192 256"}; NUM_LAYERS=${NUM_LAYERS:-"4 6"}
    ;;
  quick)
    LRS=${LRS:-"1e-4 1e-3"}; WARMUPS=${WARMUPS:-"0.0"}
    WEIGHT_DECAYS=${WEIGHT_DECAYS:-"0.0"}; SEEDS=${SEEDS:-"42"}
    EPOCH_LIST=${EPOCH_LIST:-"2"}; HIDDEN_SIZES=${HIDDEN_SIZES:-""}; NUM_LAYERS=${NUM_LAYERS:-""}
    ;;
  *)
    echo "ERROR: unknown preset '$PRESET'. Use: h3 pilot arch quick"; exit 1 ;;
esac

# Architecture dims are optional; an empty list means "leave the config's value alone".
[[ -z "$HIDDEN_SIZES" ]] && HIDDEN_SIZES="__keep__"
[[ -z "$NUM_LAYERS" ]] && NUM_LAYERS="__keep__"

PATIENCE=${PATIENCE:-5}
STOP_THRESHOLD=${STOP_THRESHOLD:-0.0001}
SCHEDULER=${SCHEDULER:-linear}

[[ -f "$BASE" ]] || { echo "ERROR: no base config: $BASE"; exit 1; }
[[ -f "$DATA_PATH" ]] || {
  echo "ERROR: no three-way tuning dataset: $DATA_PATH"
  echo "Create it with the preprocessing command in this script's header."
  exit 1
}

# --- Dataset gate: refuse to tune on a two-way file or to select the final test split ----------
"$PYTHON" - "$BASE" "$DATA_PATH" <<'PY'
import json, sys
from collections import Counter

base_path, data_path = sys.argv[1:]
with open(base_path) as f:
    base = json.load(f)
with open(data_path) as f:
    data = json.load(f)

counts = Counter(row.get("split") for row in data)
missing = {"train", "validation", "test"} - set(counts)
if missing:
    raise SystemExit(
        f"ERROR: {data_path} has split counts {dict(counts)}; missing {sorted(missing)}. "
        "Do not tune on the historical two-way train/test file."
    )
print(f"Dataset gate passed: train={counts['train']}, validation={counts['validation']}, "
      f"test={counts['test']} (test will not be selected).")
print(f"Target loop from base config: {base.get('target_region', 'h3')}")
PY
[[ $? -eq 0 ]] || exit 1

export CUDA_VISIBLE_DEVICES="$GPU"
export CDRSEQGEN_SKIP_LATEST=1          # an exploratory sweep must not steal results/runs/latest

STAMP=$(date +'%Y%m%d_%H%M%S')
OUT=${SWEEP_DIR:-"results/runs/${STAMP}_sweep_${PRESET}"}
mkdir -p "$OUT"
export CDRSEQGEN_RESULTS_ROOT="$OUT"

MAP="$OUT/runs.tsv"
SUMMARY="$OUT/summary.tsv"
if [[ ! -f "$MAP" ]]; then
  printf 'lr\twarmup\tweight_decay\tepochs\thidden\tlayers\tseed\tscheduler\tstatus\trun_dir\n' > "$MAP"
fi

total=0
for lr in $LRS; do for wu in $WARMUPS; do for wd in $WEIGHT_DECAYS; do
  for ep in $EPOCH_LIST; do for hs in $HIDDEN_SIZES; do for nl in $NUM_LAYERS; do
    for seed in $SEEDS; do total=$((total + 1)); done
done; done; done; done; done; done

echo "======================================================================"
echo "CDRSeqGen sweep | preset=$PRESET | GPU $GPU | base=$BASE | trials=$total"
echo "data=$DATA_PATH (selecting on validation)"
echo "LR=[$LRS] warmup=[$WARMUPS] weight_decay=[$WEIGHT_DECAYS] epochs=[$EPOCH_LIST] seeds=[$SEEDS]"
[[ "$HIDDEN_SIZES" != "__keep__" ]] && echo "hidden=[$HIDDEN_SIZES] layers=[$NUM_LAYERS]"
echo "patience=$PATIENCE threshold=$STOP_THRESHOLD scheduler=$SCHEDULER"
echo "output=$OUT"
echo "======================================================================"

i=0
for lr in $LRS; do
 for wu in $WARMUPS; do
  for wd in $WEIGHT_DECAYS; do
   for ep in $EPOCH_LIST; do
    for hs in $HIDDEN_SIZES; do
     for nl in $NUM_LAYERS; do
      for seed in $SEEDS; do
        i=$((i + 1))

        # Resume: a completed grid point is never re-run, so an interrupted sweep continues.
        if awk -F '\t' -v lr="$lr" -v wu="$wu" -v wd="$wd" -v ep="$ep" -v hs="$hs" -v nl="$nl" \
               -v seed="$seed" \
          'NR > 1 && $1 == lr && $2 == wu && $3 == wd && $4 == ep && $5 == hs && $6 == nl \
           && $7 == seed && $9 == "completed" {found=1} END {exit !found}' "$MAP"; then
          echo "[$i/$total] SKIP completed: lr=$lr wu=$wu wd=$wd ep=$ep hs=$hs nl=$nl seed=$seed"
          continue
        fi

        tag="lr${lr}_wu${wu}_wd${wd}_ep${ep}_s${seed}"
        [[ "$hs" != "__keep__" ]] && tag="${tag}_h${hs}l${nl}"
        export CDRSEQGEN_RUN_SUFFIX="$tag"
        LOG="$OUT/${tag}.log"

        ARCH_ARGS=()
        if [[ "$hs" != "__keep__" ]]; then
          # intermediate_size defaults to 4 x hidden_size, so the FFN ratio tracks width and the
          # comparison is not confounded by a fixed FFN. Heads must divide hidden_size.
          ARCH_ARGS=(--hidden_size "$hs" --num_layers "$nl" --num_attention_heads 8)
        fi

        echo "[$i/$total] RUN  lr=$lr wu=$wu wd=$wd ep=$ep hs=$hs nl=$nl seed=$seed"
        if "$PYTHON" train.py "$BASE" \
          --data_path "$DATA_PATH" \
          --eval_split validation \
          --learning_rate "$lr" \
          --warmup_ratio "$wu" \
          --weight_decay "$wd" \
          --num_train_epochs "$ep" \
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
          "${ARCH_ARGS[@]}" \
          > "$LOG" 2>&1; then
          RUNDIR=$(grep -oE "Workspace initialized at: .*" "$LOG" | tail -1 | sed 's/.*at: //')
          if [[ -n "$RUNDIR" && -f "$RUNDIR/epoch_metrics.csv" ]]; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tcompleted\t%s\n' \
              "$lr" "$wu" "$wd" "$ep" "$hs" "$nl" "$seed" "$SCHEDULER" "$RUNDIR" >> "$MAP"
            echo "          OK: $RUNDIR"
          else
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tincomplete\t%s\n' \
              "$lr" "$wu" "$wd" "$ep" "$hs" "$nl" "$seed" "$SCHEDULER" "${RUNDIR:-}" >> "$MAP"
            echo "          INCOMPLETE: training exited 0 but metrics/run path missing; see $LOG"
          fi
        else
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tfailed\t\n' \
            "$lr" "$wu" "$wd" "$ep" "$hs" "$nl" "$seed" "$SCHEDULER" >> "$MAP"
          echo "          FAILED: see $LOG"
        fi
      done
     done
    done
   done
  done
 done
done

# --- Rank settings by mean best-epoch validation core recovery across seeds --------------------
"$PYTHON" - "$MAP" "$SUMMARY" <<'PY'
import csv, math, os, statistics, sys
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
        trials.append({**row,
                       "best_core": float(best["eval_core_accuracy"]),
                       "best_epoch": float(best["epoch"]),
                       "final_core": float(metrics[-1]["eval_core_accuracy"])})

# Seeds are the replicate dimension: everything else identifies the setting.
groups = defaultdict(list)
for t in trials:
    groups[(t["lr"], t["warmup"], t["weight_decay"], t["epochs"], t["hidden"], t["layers"])].append(t)

ranked = []
for (lr, wu, wd, ep, hs, nl), members in groups.items():
    values = [m["best_core"] for m in members]
    ranked.append({
        "lr": lr, "warmup": wu, "weight_decay": wd, "epochs": ep,
        "hidden": hs, "layers": nl, "n_seeds": len(values),
        "mean_core": statistics.mean(values),
        "sd_core": statistics.stdev(values) if len(values) > 1 else math.nan,
        "min_core": min(values), "max_core": max(values),
        "mean_best_epoch": statistics.mean(m["best_epoch"] for m in members),
        # best - final exposes overfitting: a large gap means the curve peaked and decayed.
        "mean_decay": statistics.mean(m["best_core"] - m["final_core"] for m in members),
    })
ranked.sort(key=lambda x: (-x["mean_core"], float(x["lr"])))

fields = ["lr", "warmup", "weight_decay", "epochs", "hidden", "layers", "n_seeds",
          "mean_core", "sd_core", "min_core", "max_core", "mean_best_epoch", "mean_decay"]
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(ranked)

print("\nValidation ranking (mean best-epoch core recovery across seeds)")
print(" lr       wu     wd      ep   n   mean      sd        best_ep  decay")
for r in ranked:
    sd = "n/a" if math.isnan(r["sd_core"]) else f"{r['sd_core']:.6f}"
    arch = "" if r["hidden"] == "__keep__" else f"  h{r['hidden']}l{r['layers']}"
    print(f" {float(r['lr']):.0e}  {float(r['warmup']):>5.2f}  {float(r['weight_decay']):>5.3f}  "
          f"{float(r['epochs']):>4.0f}  {r['n_seeds']:>2}  {r['mean_core']:.6f}  {sd:>8}  "
          f"{r['mean_best_epoch']:>6.1f}  {r['mean_decay']:>+.4f}{arch}")

if ranked:
    w = ranked[0]
    print(f"\nScreening winner (validation only): lr={w['lr']}, warmup={w['warmup']}, "
          f"weight_decay={w['weight_decay']}, epochs={w['epochs']}, mean={w['mean_core']:.6f}")
    if w["n_seeds"] < 2:
        print("WARNING: winner has fewer than two completed seeds; do not freeze it yet.")
    if len(ranked) > 1:
        gap = w["mean_core"] - ranked[1]["mean_core"]
        print(f"Gap to runner-up: {gap:.6f}"
              + ("  — below the ~0.006 seed-noise floor; treat as a tie."
                 if gap < 0.006 else ""))
print(f"Summary written to {summary_path}")
PY
