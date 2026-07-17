#!/usr/bin/env bash
#
# Hyperparameter sweep for the from-scratch transformer.
#
# Why this exists: all 48 configs pin one LR per model family (scratch=1e-4, pretrained=5e-5) and
# exactly num_train_epochs=10, with no warmup / weight_decay / scheduler set — none of it ever
# swept. The h3_scratch loss is still creeping at epoch 10 (~0.001/epoch), so "10" was never
# verified as enough; meanwhile h3_esm2 peaks at epoch 4 and overfits for the remaining 6. Scratch
# runs cost ~50s, so sweeping it is cheap insurance for the "model choice doesn't matter" claim —
# scratch is the model most likely to be under-tuned.
#
# NOT swept, on purpose: hidden_size. num_attention_heads (8) and intermediate_size (512) are
# hardcoded in cdrseqgen/models/model.py and never plumbed through build_scratch_model, so
# intermediate_size does NOT follow hidden_size — changing hidden_size silently changes the FFN
# ratio too (4x at the default 128, 2x at 256), confounding the result. hidden_size must also stay
# a multiple of 8 or nn.TransformerEncoderLayer throws. Plumb those through before sweeping
# architecture.
#
# Usage:
#   ./scripts/sweep_scratch.sh <gpu> [base_config]
#   LRS="1e-4 3e-4 1e-3" EPOCHS="10 20" WARMUPS="0.0 0.06" ./scripts/sweep_scratch.sh 2
#
# Reports BEST-epoch core accuracy, not last — curves peak then decay, and the README's own rule is
# "read best-epoch". Also prints final-epoch core so overfitting is visible.

set -uo pipefail

GPU=${1:-0}
BASE=${2:-configs/h3_scratch.json}
[[ -f "$BASE" ]] || { echo "❌ No such config: $BASE"; exit 1; }

# Grid — override via env vars. Defaults probe the two open questions: is 1e-4 too low for a
# from-scratch model, and is 10 epochs enough?
LRS=${LRS:-"1e-4 3e-4 1e-3"}
EPOCHS=${EPOCHS:-"10 20"}
WARMUPS=${WARMUPS:-"0.0"}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj
export CUDA_VISIBLE_DEVICES=$GPU

STAMP=$(date +'%Y%m%d_%H%M%S')
OUT="results/sweeps/scratch_$STAMP"
mkdir -p "$OUT"
MAP="$OUT/runs.tsv"
printf 'lr\tepochs\twarmup\trun_dir\n' > "$MAP"

total=0
for lr in $LRS; do for ep in $EPOCHS; do for wu in $WARMUPS; do total=$((total + 1)); done; done; done

echo "=========================================================="
echo "🔬 scratch HP sweep  |  GPU $GPU  |  base: $BASE  |  $total runs"
echo "   LRS=[$LRS]  EPOCHS=[$EPOCHS]  WARMUPS=[$WARMUPS]"
echo "=========================================================="

i=0
for lr in $LRS; do
  for ep in $EPOCHS; do
    for wu in $WARMUPS; do
      i=$((i + 1))
      echo ""
      echo "▶️ [$i/$total] lr=$lr  epochs=$ep  warmup=$wu"
      LOG="$OUT/run_${i}.log"
      # Trailing flags override the base config (see setup_args) — the config is NOT discarded.
      python train.py "$BASE" \
        --learning_rate "$lr" \
        --num_train_epochs "$ep" \
        --warmup_ratio "$wu" \
        > "$LOG" 2>&1
      if [[ $? -ne 0 ]]; then
        echo "   ❌ FAILED — see $LOG"
        continue
      fi
      RUNDIR=$(grep -oE "Workspace initialized at: .*" "$LOG" | tail -1 | sed 's/.*at: //')
      CORE=$(python - "$RUNDIR" <<'PY'
import csv, os, sys
f = os.path.join(sys.argv[1], "epoch_metrics.csv")
rows = list(csv.DictReader(open(f))) if os.path.exists(f) else []
print(f"{max(float(r['eval_core_accuracy']) for r in rows):.4f}" if rows else "n/a")
PY
)
      echo "   ✓ best core=$CORE   $RUNDIR"
      printf '%s\t%s\t%s\t%s\n' "$lr" "$ep" "$wu" "$RUNDIR" >> "$MAP"
    done
  done
done

echo ""
echo "=========================================================="
echo "📊 Sweep summary — BEST-epoch core accuracy"
echo "=========================================================="
python - "$MAP" "$BASE" <<'PY'
import csv, json, os, sys

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
base = json.load(open(sys.argv[2]))
base_lr, base_ep = base.get("learning_rate"), base.get("num_train_epochs")

out = []
for r in rows:
    f = os.path.join(r["run_dir"], "epoch_metrics.csv")
    if not os.path.exists(f):
        continue
    m = list(csv.DictReader(open(f)))
    if not m:
        continue
    best = max(m, key=lambda x: float(x["eval_core_accuracy"]))
    out.append({
        "lr": float(r["lr"]), "ep": float(r["epochs"]), "wu": float(r["warmup"]),
        "best": float(best["eval_core_accuracy"]), "at": float(best["epoch"]),
        "final": float(m[-1]["eval_core_accuracy"]), "loss": float(best["eval_loss"]),
        "dir": r["run_dir"],
    })

if not out:
    print("  no completed runs to summarise")
    raise SystemExit

out.sort(key=lambda x: -x["best"])
print(f"{'lr':>8} {'ep':>4} {'warm':>5} {'BEST core':>10} {'@ep':>4} {'final':>8} {'loss':>8}")
for r in out:
    tag = ""
    if r["lr"] == base_lr and r["ep"] == base_ep:
        tag += "  [current default]"
    if r["best"] - r["final"] > 0.005:
        tag += "  <- overfits after peak"
    if r["at"] == r["ep"]:
        tag += "  <- still improving at last epoch (undertrained?)"
    print(f'{r["lr"]:>8.0e} {r["ep"]:>4.0f} {r["wu"]:>5.2f} {r["best"]:>10.4f} '
          f'{r["at"]:>4.0f} {r["final"]:>8.4f} {r["loss"]:>8.4f}{tag}')

b = out[0]
print()
print(f'🏆 best: lr={b["lr"]:.0e}  epochs={b["ep"]:.0f}  warmup={b["wu"]:.2f}  '
      f'-> core {b["best"]:.4f} @ epoch {b["at"]:.0f}')
print(f'   {b["dir"]}')

cur = next((r for r in out if r["lr"] == base_lr and r["ep"] == base_ep), None)
if cur and b is not cur:
    delta = b["best"] - cur["best"]
    print(f'   vs current default ({base_lr:.0e}, {base_ep:.0f}ep): {cur["best"]:.4f}  ->  {delta:+.4f}')
    if delta < 0.005:
        print("   (within noise — the ~0.32 ceiling looks tuning-robust, which SUPPORTS Finding #1)")
PY

echo ""
echo "🎉 Done. Run map: $MAP"
