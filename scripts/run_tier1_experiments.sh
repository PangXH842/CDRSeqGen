#!/usr/bin/env bash
# Coordinator for Tier 1 held-out n=1,000 experiments & CPU developability audit.
#
# Subcommands:
#   plan            CPU dry-run validating all 4 Tier 1 experiments (zero GPU)
#   developability  CPU developability/liability audit on the completed n=1,000 test benchmark
#   smoke [GPU]     Run 5-framework smoke tests on GPU (default GPU=0)
#   print-full      Print exact ready-to-launch full n=1,000 GPU commands (safe; runs nothing)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${CDR_PYTHON:-$HOME/miniconda3/envs/cdrprj/bin/python}"
IGCRAFT_PYTHON="${IGCRAFT_PYTHON:-/mnt/hdd/pang/envs/igcraft/bin/python}"
GPU="${2:-0}"

case "${1:-help}" in
  plan)
    echo "=== [1/4] Plan: MLM ESM-2 35M (2 seeds, n=1,000 test) ==="
    "$PYTHON" benchmark/run.py plan --recipe benchmark/recipes/oas4_h3_test_mlm_esm2_n1000.json --stage full
    echo "=== [2/4] Plan: IgLM-S 3-seed + Random-Init Controls (9 arms, n=1,000 test) ==="
    "$PYTHON" benchmark/run.py plan --recipe benchmark/recipes/oas4_h3_test_iglm_s_multiseed_n1000.json --stage full
    echo "=== [3/4] Plan: IgCraft T=1.0 sensitivity (n=1,000 test) ==="
    "$PYTHON" benchmark/run.py plan --recipe benchmark/recipes/oas4_h3_test_igcraft_t1_n1000.json --stage full \
      --external-python "igcraft_t1=$IGCRAFT_PYTHON"
    echo "=== [4/4] Plan: Context-Dropout confirmation (20 checkpoints x 2 regimes, n=1,000 test) ==="
    "$PYTHON" scripts/run_tier1_context_dropout_n1000.py --mode plan
    echo "All 4 Tier 1 experiment plans validated cleanly!"
    ;;

  developability)
    OUT_DIR="results/benchmark/20261124_181422_091012_oas4_h3_test_final_n1000_full"
    "$PYTHON" benchmark/analysis/developability.py \
      --batch "$OUT_DIR" \
      --batch "results/benchmark/20261126_155819_938596_oas4_h3_test_s2s_esm2_n1000_full" \
      --output-json "$OUT_DIR/developability_n1000.json" \
      --output-md "$OUT_DIR/developability_n1000.md"
    cat "$OUT_DIR/developability_n1000.md"
    ;;

  smoke)
    echo "=== Smoke [1/4]: MLM ESM-2 ==="
    "$PYTHON" benchmark/run.py smoke --recipe benchmark/recipes/oas4_h3_test_mlm_esm2_n1000.json --gpu "$GPU"
    echo "=== Smoke [2/4]: IgLM-S multiseed + controls ==="
    "$PYTHON" benchmark/run.py smoke --recipe benchmark/recipes/oas4_h3_test_iglm_s_multiseed_n1000.json --gpu "$GPU"
    echo "=== Smoke [3/4]: IgCraft T=1.0 ==="
    "$PYTHON" benchmark/run.py smoke --recipe benchmark/recipes/oas4_h3_test_igcraft_t1_n1000.json --gpu "$GPU" \
      --external-python "igcraft_t1=$IGCRAFT_PYTHON"
    echo "=== Smoke [4/4]: Context-Dropout confirmation ==="
    "$PYTHON" scripts/run_tier1_context_dropout_n1000.py --mode smoke --gpu "$GPU"
    ;;

  print-full)
    cat <<EOF
# Ready-to-launch Tier 1 full n=1,000 GPU commands (run individually or under nohup/tmux):

# 1. MLM ESM-2 35M (2 seeds, ~15 min GPU)
$PYTHON benchmark/run.py run --recipe benchmark/recipes/oas4_h3_test_mlm_esm2_n1000.json --gpu $GPU

# 2. IgLM-S 3-seed + Same-Arch Random-Init Controls (9 arms, ~45 min GPU)
$PYTHON benchmark/run.py run --recipe benchmark/recipes/oas4_h3_test_iglm_s_multiseed_n1000.json --gpu $GPU

# 3. IgCraft T=1.0 temperature sensitivity (1 arm, ~15 min GPU)
$PYTHON benchmark/run.py run --recipe benchmark/recipes/oas4_h3_test_igcraft_t1_n1000.json --gpu $GPU \\
  --external-python igcraft_t1=$IGCRAFT_PYTHON

# 4. Context-Dropout confirmation (20 checkpoints x with/without partner, partner-present IgBert judge, ~60-90 min GPU)
$PYTHON scripts/run_tier1_context_dropout_n1000.py --mode full --gpu $GPU
EOF
    ;;

  *)
    echo "Usage: $0 {plan|developability|smoke [GPU]|print-full [GPU]}" >&2
    exit 1
    ;;
esac
