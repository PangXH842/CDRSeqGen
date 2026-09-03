#!/bin/bash

# Unified experiment runner for CDRSeqGen.
#
# Usage:  ./scripts/run.sh <gpu> [suite ...]        (default suite: all)
#
# Writes one batch folder per round, named `results/runs/<timestamp>_<suites>` — e.g. `..._ar`,
# `..._oas_lcdr`, `..._all` — so a round is identifiable from `ls` alone.
#
# Suites (combine freely, e.g. `./scripts/run.sh 4 oas lcdr`):
#   oas       OAS heavy CDRs (h1/h2/h3), scratch+esm2, base + _lc            (12)
#   antigen   SAbDab2 heavy antigen matrix (_ag / _ag_lconly)                (12)
#   lcdr      OAS light CDRs (l1/l2/l3), scratch+esm2, base + _hc            (12)
#   igbert    IgBert (antibody PLM) on OAS heavy                              (6)   [~1.6GB download]
#   protbert  ProtBert (general-protein baseline) on OAS heavy               (6)   [~1.6GB download]
#   plm       igbert + protbert                                             (12)
#   ar        Autoregressive OAS heavy (h1/h2/h3), base + _lc                 (6)
#   compare   h3 masked (inferred length) + causal, one matched batch          (4)
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
# Autoregressive: same loops as the OAS suite, causal method. train.py routes on each config's
# "method" key, so nothing here needs to know which pipeline runs.
AR_CONFIGS=(
    configs/h1_ar_scratch.json configs/h1_ar_scratch_lc.json
    configs/h2_ar_scratch.json configs/h2_ar_scratch_lc.json
    configs/h3_ar_scratch.json configs/h3_ar_scratch_lc.json
)
# Paired masked/causal h3 runs in ONE batch folder, so batch_tables.md and batch_<ds>.png compare
# the two methods side by side from a single round rather than a manual merge of two.
COMPARE_CONFIGS=(
    configs/h3_scratch.json configs/h3_ar_scratch.json
    configs/h3_scratch_lc.json configs/h3_ar_scratch_lc.json
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
        ar)       printf '%s\n' "${AR_CONFIGS[@]}" ;;
        compare)  printf '%s\n' "${COMPARE_CONFIGS[@]}" ;;
        all)      printf '%s\n' "${OAS_CONFIGS[@]}" "${ANTIGEN_CONFIGS[@]}" "${LCDR_CONFIGS[@]}" ;;
        *)        printf '__UNKNOWN__:%s\n' "$1" ;;
    esac
}

# --- Resolve requested suites into a deduped, order-preserving config list ---------------------
CONFIGS=()
declare -A SEEN=()
NEEDS_SD2=0
COMPARE_REQUESTED=0
for suite in "${SUITES[@]}"; do
    [[ "$suite" == "compare" ]] && COMPARE_REQUESTED=1
    while IFS= read -r cfg; do
        if [[ "$cfg" == __UNKNOWN__:* ]]; then
            echo "❌ Unknown suite '${cfg#__UNKNOWN__:}'. Use: oas antigen lcdr ar compare igbert protbert plm all"
            exit 1
        fi
        [[ -n "${SEEN[$cfg]}" ]] && continue
        SEEN[$cfg]=1
        CONFIGS+=("$cfg")
        [[ "$cfg" == *_ag*.json ]] && NEEDS_SD2=1
    done < <(expand_suite "$suite")
done

# --- One folder per round ---------------------------------------------------------------------
# Named `<stamp>_<suites>`, the same `<timestamp>_<name>` convention sweep.sh uses for
# `<stamp>_sweep_<preset>`. A bare timestamp said nothing about what a round contained, so telling
# an `ar` batch from an `oas` one meant opening a run_config.json; the suites are already validated
# above, so the tag can only contain known names.
BATCH_STAMP=$(date +'%Y%m%d_%H%M%S')
SUITE_TAG=""
declare -A SUITE_SEEN=()
for suite in "${SUITES[@]}"; do
    [[ -n "${SUITE_SEEN[$suite]}" ]] && continue
    SUITE_SEEN[$suite]=1
    SUITE_TAG+="_$suite"
done
BATCH_DIR="results/runs/${BATCH_STAMP}${SUITE_TAG}"
mkdir -p "$BATCH_DIR"
export CDRSEQGEN_RESULTS_ROOT="$BATCH_DIR"

echo "=========================================================="
echo "🧬 CDRSeqGen runner  |  GPU $GPU_ID  |  suites: ${SUITES[*]}  |  ${#CONFIGS[@]} runs"
echo "📦 Batch dir: $BATCH_DIR"
echo "=========================================================="

# --- Fail fast if the processed data a suite needs isn't there yet -----------------------------
# Gate on the file the configs actually declare, not on a fixed name: every OAS config now points
# at processed_oas_tuning.json (three-way), so checking processed_oas_data.json passed while telling
# you nothing about whether the run could start.
for cfg in "${CONFIGS[@]}"; do
    need=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('data_path',''))" "$cfg")
    if [[ -n "$need" && ! -f "$need" ]]; then
        echo "❌ $cfg needs $need, which does not exist."
        echo "   Create it with the preprocessing commands in WORKFLOW.md Step 1."
        exit 1
    fi
done
if [[ $NEEDS_SD2 -eq 1 && ! -f dataset/data/processed/processed_sabdab2_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_sabdab2_data.json — run:"
    echo "     bash cdrseqgen/data/preprocess/download_sabdab2.sh && python -m cdrseqgen.data.preprocess.sabdab2"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

SUCCESS_COUNT=0
FAILED_COUNT=0
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] Starting run: $config"
    echo "----------------------------------------------------------"
    RUN_OVERRIDES=()
    if [[ $COMPARE_REQUESTED -eq 1 &&
          ( "$config" == "configs/h3_scratch.json" ||
            "$config" == "configs/h3_scratch_lc.json" ) ]]; then
        RUN_OVERRIDES=(--gen_length_mode inferred)
        echo "⚖️  Matched compare: MLM generation uses inferred length (no native mask count)."
    fi
    if python train.py "$config" "${RUN_OVERRIDES[@]}"; then
        ((SUCCESS_COUNT+=1))
        echo "✅ Finished run: $config"
    else
        ((FAILED_COUNT+=1))
        echo "❌ WARNING: Run failed and was skipped: $config"
    fi
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
echo "🎉 Done ($SUCCESS_COUNT/${#CONFIGS[@]} succeeded; $FAILED_COUNT failed). Artifacts are in $BATCH_DIR/"
echo "   $(find "$BATCH_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l) run dir(s), plus batch_tables.md and figures/ (one dashboard per dataset)."
echo "   Ad-hoc cross-run charts: 'python -m cdrseqgen.visualization.linegraph_plotter'."
