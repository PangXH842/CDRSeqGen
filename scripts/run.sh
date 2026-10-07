#!/usr/bin/env bash
# One launcher for named experiments and legacy suite/sweep commands.
#   ./scripts/run.sh list
#   ./scripts/run.sh plan mlm_no_partner_controls --gpus 0
#   ./scripts/run.sh run mlm_no_partner_controls --gpus 0
#   ./scripts/run.sh status results/runs/<batch>
#   ./scripts/run.sh resume results/runs/<batch> --gpus 0
# See --help and configs/experiments/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate cdrprj
fi
case "${1:-}" in
    list|plan|run|status|resume|--help|-h|"")
        exec "${PYTHON:-python}" scripts/experiments.py "$@"
        ;;
    *)
        exec "${PYTHON:-python}" -m cdrseqgen.experiments "$@"
        ;;
esac
