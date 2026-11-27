#!/usr/bin/env bash
# Create fresh environments from requirements/*.txt; never update an existing research environment in place.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_DIR="$ROOT/requirements"

AVAILABLE=()
for req in "$REQ_DIR"/*.txt; do
    [[ -f "$req" ]] || continue
    AVAILABLE+=("$(basename "$req" .txt)")
done
if [[ ${#AVAILABLE[@]} -eq 0 ]]; then
    echo "No requirements/*.txt files found in $REQ_DIR" >&2
    exit 1
fi

usage() {
    local joined
    joined="$(IFS='|'; echo "${AVAILABLE[*]}")"
    echo "Usage: bash env.sh [all|list|$joined]"
}

TARGET="${1:-all}"
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "$TARGET" in
    -h|--help)
        usage
        exit 0
        ;;
    list)
        printf '%s\n' "${AVAILABLE[@]}"
        exit 0
        ;;
    all)
        ENVS=("${AVAILABLE[@]}")
        ;;
    *)
        if [[ -f "$REQ_DIR/$TARGET.txt" ]]; then
            ENVS=("$TARGET")
        else
            echo "Unknown environment '$TARGET' (no $REQ_DIR/$TARGET.txt)." >&2
            usage >&2
            exit 2
        fi
        ;;
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
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$REQ_DIR/$ENV_NAME.txt"
    conda run --no-capture-output -n "$ENV_NAME" python -m pip check
done
echo "Setup complete. Run: bash $ROOT/scripts/verify_env.sh $TARGET"
