#!/usr/bin/env bash
# One launcher for named experiments and legacy suite/sweep commands.
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
# See --help and configs/experiments/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${PYTHON:-}" && -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate cdrprj
fi
case "${1:-}" in
    benchmark)
        shift
        if [[ "${1:-}" == "followup" ]]; then
            shift
            exec "${PYTHON:-python}" benchmark/followup.py "$@"
        fi
        if [[ "${1:-}" == "variability" ]]; then
            shift
            exec "${PYTHON:-python}" benchmark/variability.py "$@"
        fi
        if [[ "${1:-}" == "extend" ]]; then
            shift
            exec "${PYTHON:-python}" benchmark/extend.py "$@"
        fi
        if [[ "${1:-}" == "leakage" ]]; then
            shift
            exec "${PYTHON:-python}" benchmark/leakage.py "$@"
        fi
        exec "${PYTHON:-python}" benchmark/run.py "$@"
        ;;
    list|plan|run|status|resume|--help|-h|"")
        exec "${PYTHON:-python}" scripts/experiments.py "$@"
        ;;
    *)
        exec "${PYTHON:-python}" -m cdrseqgen.experiments "$@"
        ;;
esac
