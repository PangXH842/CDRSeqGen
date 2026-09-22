#!/usr/bin/env bash
# One launcher for suites and sweeps; usable from any working directory.
#   ./scripts/run.sh 0 compare
#   ./scripts/run.sh 0,1 sweep lr configs/s2s/h3_s2s_scratch.json
#   DRY_RUN=1 ./scripts/run.sh 0 compare
# See --help and WORKFLOW.md for presets and overrides.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate cdrprj
fi
exec "${PYTHON:-python}" -m cdrseqgen.experiments "$@"
