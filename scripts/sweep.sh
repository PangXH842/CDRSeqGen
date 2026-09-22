#!/usr/bin/env bash
# Compatibility alias for: run.sh <gpus> sweep [preset] [base_config]
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    exec bash "$(dirname "$0")/run.sh" --help
fi
if [[ $# -gt 3 ]]; then
    echo "Usage: sweep.sh <gpu>[,<gpu>...] [preset] [base_config]" >&2
    exit 1
fi
exec bash "$(dirname "$0")/run.sh" "${1:-0}" sweep "${2:-h3}" "${3:-configs/mlm/heavy/h3_scratch.json}"
