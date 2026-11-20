#!/usr/bin/env bash
# One launcher for named experiments, legacy suite/sweep commands and the benchmark tools: python -m experiments.
#   ./scripts/run.sh list
#   ./scripts/run.sh plan mlm_no_partner_controls --gpus 0
#   ./scripts/run.sh run mlm_no_partner_controls --gpus 0
#   ./scripts/run.sh status results/runs/<batch>
#   ./scripts/run.sh resume results/runs/<batch> --gpus 0
#   ./scripts/run.sh benchmark plan --stage smoke
#   ./scripts/run.sh benchmark smoke --gpu 0
#   ./scripts/run.sh benchmark followup plan
#   ./scripts/run.sh benchmark followup run --gpu 0
#   ./scripts/run.sh benchmark extend plan --added results/benchmark/<batch of added externals>
#   ./scripts/run.sh benchmark variability run [--references results/benchmark/<retrieval batch>]
#   ./scripts/run.sh benchmark leakage plan
#   ./scripts/run.sh benchmark leakage run --output results/benchmark/<audit>
# The first argument picks the interface (experiments/__main__.py). See --help and experiments/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate cdrprj
fi
exec "${PYTHON:-python}" -m experiments "$@"
