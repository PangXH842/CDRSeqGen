#!/bin/bash

# Unified experiment runner for CDRSeqGen.
#
# Usage:  ./scripts/run.sh <gpu> [suite ...]        (default suite: all)
#
# Suites (combine freely, e.g. `./scripts/run.sh 4 oas lcdr`):
#   oas       OAS heavy CDRs (h1/h2/h3), scratch+esm2, base + _lc            (12)
#   antigen   SAbDab2 heavy antigen matrix (_ag / _ag_lconly)                (12)
#   lcdr      OAS light CDRs (l1/l2/l3), scratch+esm2, base + _hc            (12)
#   igbert    IgBert (antibody PLM) on OAS heavy                              (6)   [~1.6GB download]
#   protbert  ProtBert (general-protein baseline) on OAS heavy               (6)   [~1.6GB download]
#   plm       igbert + protbert                                             (12)
#   all       oas + antigen + lcdr (the scratch/esm2 matrices; PLMs opt-in)  (36)
#
# Scratch runs come first within each suite (cheap — a broken config fails fast before ESM-2/PLMs);
# a failed run is logged and skipped. OAS suites use whatever split the processed JSON carries
# (clustered by default; see cdrseqgen/data/preprocess/oas_paired.py).

GPU_ID=${1:-0}
shift 2>/dev/null
SUITES=("$@")
[[ ${#SUITES[@]} -eq 0 ]] && SUITES=("all")

# Activate the environment (Adjust the path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj

# --- Suite definitions (scratch first within each) --------------------------------------------
OAS_CONFIGS=(
    configs/h1_scratch.json configs/h1_scratch_lc.json
    configs/h2_scratch.json configs/h2_scratch_lc.json
    configs/h3_scratch.json configs/h3_scratch_lc.json
    configs/h1_esm2.json configs/h1_esm2_lc.json
    configs/h2_esm2.json configs/h2_esm2_lc.json
    configs/h3_esm2.json configs/h3_esm2_lc.json
)
# _ag = SAbDab2 + light + antigen; _ag_lconly = matched baseline (same data, antigen OFF).
ANTIGEN_CONFIGS=(
    configs/h1_scratch_ag_lconly.json configs/h2_scratch_ag_lconly.json configs/h3_scratch_ag_lconly.json
    configs/h1_scratch_ag.json configs/h2_scratch_ag.json configs/h3_scratch_ag.json
    configs/h1_esm2_ag_lconly.json configs/h2_esm2_ag_lconly.json configs/h3_esm2_ag_lconly.json
    configs/h1_esm2_ag.json configs/h2_esm2_ag.json configs/h3_esm2_ag.json
)
# Light CDRs: partner context is the HEAVY chain (_hc), the mirror of _lc for heavy targets.
LCDR_CONFIGS=(
    configs/l1_scratch.json configs/l1_scratch_hc.json
    configs/l2_scratch.json configs/l2_scratch_hc.json
    configs/l3_scratch.json configs/l3_scratch_hc.json
    configs/l1_esm2.json configs/l1_esm2_hc.json
    configs/l2_esm2.json configs/l2_esm2_hc.json
    configs/l3_esm2.json configs/l3_esm2_hc.json
)
IGBERT_CONFIGS=(
    configs/h1_igbert.json configs/h1_igbert_lc.json
    configs/h2_igbert.json configs/h2_igbert_lc.json
    configs/h3_igbert.json configs/h3_igbert_lc.json
)
PROTBERT_CONFIGS=(
    configs/h1_protbert.json configs/h1_protbert_lc.json
    configs/h2_protbert.json configs/h2_protbert_lc.json
    configs/h3_protbert.json configs/h3_protbert_lc.json
)

expand_suite() {
    case "$1" in
        oas)      printf '%s\n' "${OAS_CONFIGS[@]}" ;;
        antigen)  printf '%s\n' "${ANTIGEN_CONFIGS[@]}" ;;
        lcdr)     printf '%s\n' "${LCDR_CONFIGS[@]}" ;;
        igbert)   printf '%s\n' "${IGBERT_CONFIGS[@]}" ;;
        protbert) printf '%s\n' "${PROTBERT_CONFIGS[@]}" ;;
        plm)      printf '%s\n' "${IGBERT_CONFIGS[@]}" "${PROTBERT_CONFIGS[@]}" ;;
        all)      printf '%s\n' "${OAS_CONFIGS[@]}" "${ANTIGEN_CONFIGS[@]}" "${LCDR_CONFIGS[@]}" ;;
        *)        printf '__UNKNOWN__:%s\n' "$1" ;;
    esac
}

# --- Resolve requested suites into a deduped, order-preserving config list ---------------------
CONFIGS=()
declare -A SEEN=()
NEEDS_OAS=0
NEEDS_SD2=0
for suite in "${SUITES[@]}"; do
    while IFS= read -r cfg; do
        if [[ "$cfg" == __UNKNOWN__:* ]]; then
            echo "❌ Unknown suite '${cfg#__UNKNOWN__:}'. Use: oas antigen lcdr igbert protbert plm all"
            exit 1
        fi
        [[ -n "${SEEN[$cfg]}" ]] && continue
        SEEN[$cfg]=1
        CONFIGS+=("$cfg")
        if [[ "$cfg" == *_ag*.json ]]; then NEEDS_SD2=1; else NEEDS_OAS=1; fi
    done < <(expand_suite "$suite")
done

# --- One folder per round ---------------------------------------------------------------------
BATCH_STAMP=$(date +'%Y%m%d_%H%M%S')
BATCH_DIR="results/runs/$BATCH_STAMP"
mkdir -p "$BATCH_DIR"
export CDRSEQGEN_RESULTS_ROOT="$BATCH_DIR"

echo "=========================================================="
echo "🧬 CDRSeqGen runner  |  GPU $GPU_ID  |  suites: ${SUITES[*]}  |  ${#CONFIGS[@]} runs"
echo "📦 Batch dir: $BATCH_DIR"
echo "=========================================================="

# --- Fail fast if the processed data a suite needs isn't there yet -----------------------------
if [[ $NEEDS_OAS -eq 1 && ! -f dataset/data/processed/processed_oas_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_oas_data.json — run:"
    echo "     python -m cdrseqgen.data.preprocess.oas_paired"
    exit 1
fi
if [[ $NEEDS_SD2 -eq 1 && ! -f dataset/data/processed/processed_sabdab2_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_sabdab2_data.json — run:"
    echo "     bash cdrseqgen/data/preprocess/download_sabdab2.sh && python -m cdrseqgen.data.preprocess.sabdab2"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] Starting run: $config"
    echo "----------------------------------------------------------"
    # || true so a crashed run doesn't abort the whole matrix.
    python train.py "$config" || { echo "❌ WARNING: Run failed for $config"; true; }
    echo "✅ Finished run: $config"
done

# --- Reporting: comparison tables + dashboards, written beside the runs ------------------------
# Each run already drops its own curves next to its epoch_metrics.csv; these are the batch-level
# equivalent, so a finished round is self-describing without a second command. Both steps are
# guarded: the matrix has already trained, and a reporting failure must not be able to spoil it.
echo ""
echo "📊 Generating batch tables and dashboards..."
python -m cdrseqgen.visualization.batch_tables --batch "$BATCH_DIR" --output \
  || echo "⚠️ Table step failed; the runs are intact — rerun it against $BATCH_DIR"
python -m cdrseqgen.visualization.batch_plots --batch "$BATCH_DIR" \
  || echo "⚠️ Plot step failed; the runs are intact — rerun it against $BATCH_DIR"

echo ""
echo "🎉 Done (${#CONFIGS[@]} runs). Everything for this batch is in $BATCH_DIR/"
echo "   $(find "$BATCH_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l) run dir(s), plus batch_tables.md and figures/ (one dashboard per dataset)."
echo "   Ad-hoc cross-run charts: 'python -m cdrseqgen.visualization.linegraph_plotter'."
