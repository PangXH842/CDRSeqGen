#!/usr/bin/env bash
# GPU validation by default; --cpu explicitly selects import/CPU validation only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$TARGET" in
    all) ENVS=(cdrprj iglm) ;;
    cdrprj|iglm) ENVS=("$TARGET") ;;
    -h|--help) echo "Usage: bash scripts/verify_env.sh [all|cdrprj|iglm] [--cpu]"; exit 0 ;;
    *) echo "Usage: bash scripts/verify_env.sh [all|cdrprj|iglm] [--cpu]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --cpu ) ]]; then
    echo "Only --cpu is accepted after the environment name." >&2
    exit 2
fi
for ENV_NAME in "${ENVS[@]}"; do
    conda run --no-capture-output -n "$ENV_NAME" python -m pip check
    conda run --no-capture-output -n "$ENV_NAME" python scripts/check_environment.py "$ENV_NAME" "$@"
done
echo "All requested environment checks passed."
