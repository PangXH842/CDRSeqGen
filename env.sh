#!/usr/bin/env bash
# Create fresh environments; never update an existing research environment in place.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-all}"
if [[ $# -gt 1 ]]; then
    echo "Usage: bash env.sh [all|cdrprj|iglm]" >&2
    exit 2
fi
case "$TARGET" in
    all) ENVS=(cdrprj iglm) ;;
    cdrprj|iglm) ENVS=("$TARGET") ;;
    -h|--help) echo "Usage: bash env.sh [all|cdrprj|iglm]"; exit 0 ;;
    *) echo "Usage: bash env.sh [all|cdrprj|iglm]" >&2; exit 2 ;;
esac
command -v conda >/dev/null || { echo "Conda must be on PATH." >&2; exit 1; }
# Check every requested name before creating anything. Never let -y replace an env.
CONDA_BASE="$(conda info --base)"
conda env list --json | "$CONDA_BASE/bin/python" -c '
import json, pathlib, sys
existing = {pathlib.Path(p).name for p in json.load(sys.stdin)["envs"]}
conflicts = existing.intersection(sys.argv[1:])
if conflicts:
    sys.exit("Refusing to replace existing environments: " + ", ".join(sorted(conflicts)))
' "${ENVS[@]}"
for ENV_NAME in "${ENVS[@]}"; do
    conda create --name "$ENV_NAME" python=3.11 pip -y
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$ROOT/requirements/$ENV_NAME.txt"
    conda run --no-capture-output -n "$ENV_NAME" python -m pip check
done
echo "Setup complete. Run: bash $ROOT/scripts/verify_env.sh $TARGET"
