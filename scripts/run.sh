#!/bin/bash

# Unified experiment runner for CDRSeqGen.
#
# Usage:  ./scripts/run.sh <gpu>[,<gpu>...] [suite ...]        (default suite: all)
#
# Several GPUs shard the round's runs round-robin, one process per device — the runs of a batch are
# independent, so that is where the parallelism is (no DDP inside a single training). Per-GPU output
# goes to <batch>/logs/gpu<id>.log instead of the terminal; a single GPU still streams as before.
#   ./scripts/run.sh 2 ar_heavy                 6 runs on GPU 2, sequential
#   ./scripts/run.sh 2,5,6 mlm_heavy ar_heavy   18 runs across three GPUs (~3x faster)
#
# Writes one batch folder per round, named `results/runs/<timestamp>_<suites>` — e.g. `..._ar`,
# `..._oas_lcdr`, `..._all` — so a round is identifiable from `ls` alone.
#
# DATA_PATH=<processed.json> ./scripts/run.sh <gpu> <suite>   runs a suite against a DIFFERENT
# corpus without duplicating configs, the same override sweep.sh takes. The run name picks up that
# corpus's dataset tag (dataset_tag in cdrseqgen/train/logging.py), so runs on two corpora cannot
# collide in one batch table — which is the whole reason the override is safe to offer.
#
# Suites (combine freely, e.g. `./scripts/run.sh 4 mlm_heavy ar_heavy`). Each names its METHOD
# first, then what varies within it:
#   mlm_heavy     masked, OAS heavy CDRs (h1/h2/h3), scratch+esm2, base + _lc    (12)
#   mlm_light     masked, OAS light CDRs (l1/l2/l3), scratch+esm2, base + _hc    (12)
#   mlm_antigen   masked, SAbDab2 heavy antigen matrix (_ag / _ag_lconly)        (12)
#   mlm_igbert    masked, IgBert (antibody PLM) on OAS heavy                      (6)  [~1.6GB dl]
#   mlm_protbert  masked, ProtBert (general-protein baseline) on OAS heavy        (6)  [~1.6GB dl]
#   mlm_plm       mlm_igbert + mlm_protbert                                      (12)
#   ar_heavy      causal, OAS heavy CDRs (h1/h2/h3), base + _lc                   (6)
#   compare       h3 masked (inferred length) + causal, one matched batch         (4)
#   all           mlm_heavy + mlm_antigen + mlm_light (PLMs and causal opt-in)   (36)
#
# Renamed 2026-09-03 (hard rename, old names error with the new spelling): oas -> mlm_heavy,
# lcdr -> mlm_light, ar -> ar_heavy, antigen -> mlm_antigen, igbert/protbert/plm -> mlm_*.
#
# Scratch runs come first within each suite (cheap — a broken config fails fast before ESM-2/PLMs);
# a failed run is logged and skipped. OAS suites use whatever split the processed JSON carries
# (clustered by default; see cdrseqgen/data/preprocess/oas_paired.py).

# One GPU, or several: "2", "2,5,6" and "2 5 6" are all accepted. With more than one, the runs of
# a round are sharded across them round-robin -- runs are fully independent, so this is where the
# parallelism is, not inside a single training.
GPU_ARG=${1:-0}
shift 2>/dev/null
IFS=', ' read -r -a GPUS <<< "$GPU_ARG"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "❌ '$gpu' is not a GPU index. Use e.g. 0, or 2,5,6"; exit 1; }
done
GPU_ID="${GPUS[0]}"
SUITES=("$@")
[[ ${#SUITES[@]} -eq 0 ]] && SUITES=("all")
# Empty means "whatever each config declares"; set it to run a suite on another corpus.
DATA_PATH=${DATA_PATH:-}

# Activate the environment (Adjust the path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj

# --- Suite definitions (scratch first within each) --------------------------------------------
MLM_HEAVY_CONFIGS=(
    configs/h1_scratch.json configs/h1_scratch_lc.json
    configs/h2_scratch.json configs/h2_scratch_lc.json
    configs/h3_scratch.json configs/h3_scratch_lc.json
    configs/h1_esm2.json configs/h1_esm2_lc.json
    configs/h2_esm2.json configs/h2_esm2_lc.json
    configs/h3_esm2.json configs/h3_esm2_lc.json
)
# _ag = SAbDab2 + light + antigen; _ag_lconly = matched baseline (same data, antigen OFF).
MLM_ANTIGEN_CONFIGS=(
    configs/h1_scratch_ag_lconly.json configs/h2_scratch_ag_lconly.json configs/h3_scratch_ag_lconly.json
    configs/h1_scratch_ag.json configs/h2_scratch_ag.json configs/h3_scratch_ag.json
    configs/h1_esm2_ag_lconly.json configs/h2_esm2_ag_lconly.json configs/h3_esm2_ag_lconly.json
    configs/h1_esm2_ag.json configs/h2_esm2_ag.json configs/h3_esm2_ag.json
)
# Light CDRs: partner context is the HEAVY chain (_hc), the mirror of _lc for heavy targets.
MLM_LIGHT_CONFIGS=(
    configs/l1_scratch.json configs/l1_scratch_hc.json
    configs/l2_scratch.json configs/l2_scratch_hc.json
    configs/l3_scratch.json configs/l3_scratch_hc.json
    configs/l1_esm2.json configs/l1_esm2_hc.json
    configs/l2_esm2.json configs/l2_esm2_hc.json
    configs/l3_esm2.json configs/l3_esm2_hc.json
)
# Autoregressive: same loops as the OAS suite, causal method. train.py routes on each config's
# "method" key, so nothing here needs to know which pipeline runs.
AR_HEAVY_CONFIGS=(
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
MLM_IGBERT_CONFIGS=(
    configs/h1_igbert.json configs/h1_igbert_lc.json
    configs/h2_igbert.json configs/h2_igbert_lc.json
    configs/h3_igbert.json configs/h3_igbert_lc.json
)
MLM_PROTBERT_CONFIGS=(
    configs/h1_protbert.json configs/h1_protbert_lc.json
    configs/h2_protbert.json configs/h2_protbert_lc.json
    configs/h3_protbert.json configs/h3_protbert_lc.json
)

# Every suite names its METHOD first, then what varies within it. The old names picked a different
# axis each time -- `oas` a dataset (but lcdr/ar/igbert/protbert are equally OAS), `lcdr` a loop
# set, `ar` a method, `igbert` a model, `antigen` a context -- which stopped being merely untidy
# once suite names became part of the batch folder name and `dataset_tag` grew a second OAS corpus.
SUITE_NAMES="mlm_heavy mlm_light mlm_antigen mlm_igbert mlm_protbert mlm_plm ar_heavy compare all"

expand_suite() {
    case "$1" in
        mlm_heavy)    printf '%s\n' "${MLM_HEAVY_CONFIGS[@]}" ;;
        mlm_light)    printf '%s\n' "${MLM_LIGHT_CONFIGS[@]}" ;;
        mlm_antigen)  printf '%s\n' "${MLM_ANTIGEN_CONFIGS[@]}" ;;
        mlm_igbert)   printf '%s\n' "${MLM_IGBERT_CONFIGS[@]}" ;;
        mlm_protbert) printf '%s\n' "${MLM_PROTBERT_CONFIGS[@]}" ;;
        mlm_plm)      printf '%s\n' "${MLM_IGBERT_CONFIGS[@]}" "${MLM_PROTBERT_CONFIGS[@]}" ;;
        ar_heavy)     printf '%s\n' "${AR_HEAVY_CONFIGS[@]}" ;;
        compare)      printf '%s\n' "${COMPARE_CONFIGS[@]}" ;;
        all)          printf '%s\n' "${MLM_HEAVY_CONFIGS[@]}" "${MLM_ANTIGEN_CONFIGS[@]}" "${MLM_LIGHT_CONFIGS[@]}" ;;
        *)            printf '__UNKNOWN__:%s\n' "$1" ;;
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
            bad="${cfg#__UNKNOWN__:}"
            echo "❌ Unknown suite '$bad'. Use: $SUITE_NAMES"
            case "$bad" in
                oas)      echo "   (renamed 2026-09-03: oas -> mlm_heavy)" ;;
                lcdr)     echo "   (renamed 2026-09-03: lcdr -> mlm_light)" ;;
                ar)       echo "   (renamed 2026-09-03: ar -> ar_heavy)" ;;
                antigen)  echo "   (renamed 2026-09-03: antigen -> mlm_antigen)" ;;
                igbert)   echo "   (renamed 2026-09-03: igbert -> mlm_igbert)" ;;
                protbert) echo "   (renamed 2026-09-03: protbert -> mlm_protbert)" ;;
                plm)      echo "   (renamed 2026-09-03: plm -> mlm_plm)" ;;
            esac
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
echo "🧬 CDRSeqGen runner  |  GPU(s) ${GPUS[*]}  |  suites: ${SUITES[*]}  |  ${#CONFIGS[@]} runs"
echo "📦 Batch dir: $BATCH_DIR"
echo "=========================================================="

# --- Fail fast if the processed data a suite needs isn't there yet -----------------------------
# Gate on the file the configs actually declare, not on a fixed name: every OAS config now points
# at processed_oas_tuning.json (three-way), so checking processed_oas_data.json passed while telling
# you nothing about whether the run could start.
if [[ -n "$DATA_PATH" ]]; then
    if [[ ! -f "$DATA_PATH" ]]; then
        echo "❌ DATA_PATH=$DATA_PATH does not exist."
        echo "   Create it with the preprocessing commands in WORKFLOW.md Step 1."
        exit 1
    fi
    # DATA_PATH replaces EVERY config's corpus, including one built for a different dataset. A
    # SAbDab2 `_ag` config forced onto OAS does not fail: dataset_common reads the antigen as
    # `item.get("antigen") or {}`, so an OAS entry (which has no antigen key at all) silently
    # yields no antigen context. The run completes, looks healthy, and quietly turns the
    # `_ag` vs `_ag_lconly` pair into two identical arms. Refuse instead.
    if grep -lE '"include_antigen"[[:space:]]*:[[:space:]]*true' "${CONFIGS[@]}" >/dev/null 2>&1; then
        if ! python -c "
import json, sys
entries = json.load(open(sys.argv[1]))
sys.exit(0 if any(e.get('antigen') for e in entries[:1000]) else 1)
" "$DATA_PATH"; then
            echo "❌ $DATA_PATH carries no antigen data, but these configs need it:"
            grep -lE '"include_antigen"[[:space:]]*:[[:space:]]*true' "${CONFIGS[@]}" | sed 's/^/     /'
            echo "   They would train with the antigen silently absent. Drop the 'antigen' suite,"
            echo "   or run it separately without DATA_PATH."
            exit 1
        fi
    fi
    echo "📄 Corpus override: every run uses $DATA_PATH (not the path its config declares)."
else
    for cfg in "${CONFIGS[@]}"; do
        need=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('data_path',''))" "$cfg")
        if [[ -n "$need" && ! -f "$need" ]]; then
            echo "❌ $cfg needs $need, which does not exist."
            echo "   Create it with the preprocessing commands in WORKFLOW.md Step 1."
            exit 1
        fi
    done
fi
if [[ $NEEDS_SD2 -eq 1 && ! -f dataset/data/processed/processed_sabdab2_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_sabdab2_data.json — run:"
    echo "     bash cdrseqgen/data/preprocess/download_sabdab2.sh && python -m cdrseqgen.data.preprocess.sabdab2"
    exit 1
fi

# One place that knows how to invoke a run, so the sequential and parallel drivers below cannot
# drift on the compare overrides or the corpus override.
run_one() {
    local gpu="$1" config="$2"
    local overrides=()
    [[ -n "$DATA_PATH" ]] && overrides+=(--data_path "$DATA_PATH")
    if [[ $COMPARE_REQUESTED -eq 1 &&
          ( "$config" == "configs/h3_scratch.json" ||
            "$config" == "configs/h3_scratch_lc.json" ) ]]; then
        # Three things must match for a masked-vs-causal head-to-head, and only the first was
        # being forced. The other two are config defaults that happen to differ:
        #   max_len   -- the causal layout needs 384 (framework + partner chain + the 96-position
        #                answer reserve + 3 layout tokens), while h3_scratch.json ships 128. 12% of
        #                OAS heavy chains exceed 128 tokens, so the masked arm was seeing TRUNCATED
        #                flanks on those examples while the causal arm saw complete ones. Raised
        #                from 256 on 2026-09-04 with the AR configs, so the pair stays matched.
        #   gen_top_p -- the AR configs pin 0.9; the masked default is 1.0, i.e. no nucleus
        #                filtering at all. Different sampling policies are not a comparison.
        overrides+=(--gen_length_mode inferred --max_len 384 --gen_top_p 0.9)
        echo "⚖️  Matched compare: MLM uses inferred length, max_len 384, top_p 0.9 (matching AR)."
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python train.py "$config" "${overrides[@]}"
}

SUCCESS_COUNT=0
FAILED_COUNT=0

if [[ ${#GPUS[@]} -eq 1 ]]; then
    for config in "${CONFIGS[@]}"; do
        echo ""
        echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] Starting run: $config"
        echo "----------------------------------------------------------"
        if run_one "${GPUS[0]}" "$config"; then
            ((SUCCESS_COUNT+=1))
            echo "✅ Finished run: $config"
        else
            ((FAILED_COUNT+=1))
            echo "❌ WARNING: Run failed and was skipped: $config"
        fi
    done
else
    # Round-robin, so the expensive configs spread out: a suite lists cheap scratch runs first and
    # ESM-2/PLM runs after, and striding by GPU count hands each device a mix of both rather than
    # one device the whole slow tail.
    mkdir -p "$BATCH_DIR/logs"
    # Every shard would otherwise race to write results/runs/latest at the same target; write it
    # once here instead.
    export CDRSEQGEN_SKIP_LATEST=1
    echo ""
    echo "🧵 Sharding ${#CONFIGS[@]} runs across ${#GPUS[@]} GPUs (round-robin):"
    for ((i = 0; i < ${#GPUS[@]}; i++)); do
        shard=()
        for ((j = i; j < ${#CONFIGS[@]}; j += ${#GPUS[@]})); do shard+=("$(basename "${CONFIGS[j]}" .json)"); done
        echo "   gpu ${GPUS[i]}: ${#shard[@]} runs — ${shard[*]}"
    done
    echo "   Per-GPU output: $BATCH_DIR/logs/gpu<id>.log"

    for ((i = 0; i < ${#GPUS[@]}; i++)); do
        (
            for ((j = i; j < ${#CONFIGS[@]}; j += ${#GPUS[@]})); do
                config="${CONFIGS[j]}"
                echo ""
                echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] gpu ${GPUS[i]} starting: $config"
                echo "----------------------------------------------------------"
                if run_one "${GPUS[i]}" "$config"; then
                    echo "ok $config" >> "$BATCH_DIR/.status.${GPUS[i]}"
                    echo "✅ Finished run: $config"
                else
                    echo "fail $config" >> "$BATCH_DIR/.status.${GPUS[i]}"
                    echo "❌ WARNING: Run failed and was skipped: $config"
                fi
            done
        ) > "$BATCH_DIR/logs/gpu${GPUS[i]}.log" 2>&1 &
    done
    wait

    SUCCESS_COUNT=$(cat "$BATCH_DIR"/.status.* 2>/dev/null | grep -c '^ok ' || true)
    FAILED_COUNT=$(cat "$BATCH_DIR"/.status.* 2>/dev/null | grep -c '^fail ' || true)
    if [[ $FAILED_COUNT -gt 0 ]]; then
        echo ""
        echo "Failed runs (see $BATCH_DIR/logs/):"
        cat "$BATCH_DIR"/.status.* 2>/dev/null | sed -n 's/^fail /   ❌ /p'
    fi
    rm -f "$BATCH_DIR"/.status.*
    # The link the shards were told to skip, written once now that the round is done.
    ln -sfn "$(basename "$BATCH_DIR")" "results/runs/latest" 2>/dev/null || true
fi

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
if [[ $FAILED_COUNT -gt 0 ]]; then
    echo "❌ Done with failures ($SUCCESS_COUNT/${#CONFIGS[@]} succeeded; $FAILED_COUNT failed). Artifacts are in $BATCH_DIR/"
else
    echo "🎉 Done ($SUCCESS_COUNT/${#CONFIGS[@]} succeeded). Artifacts are in $BATCH_DIR/"
fi
echo "   $(find "$BATCH_DIR" -maxdepth 1 -mindepth 1 -type d | wc -l) run dir(s), plus batch_tables.md and figures/ (one dashboard per dataset)."
echo "   Ad-hoc cross-run charts: 'python -m cdrseqgen.visualization.linegraph_plotter'."
[[ $FAILED_COUNT -eq 0 ]] || exit 1
