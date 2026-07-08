#!/bin/bash

# Default to GPU 0 if the user doesn't provide one
GPU_ID=${1:-0}
# Suite selector (default "all" = oas + antigen). Options:
#   oas | antigen | all | igbert | protbert | plm   (plm = igbert + protbert; both download ~1.6GB)
SUITE=${2:-all}

# Activate the environment (Adjust the path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj

echo "=========================================================="
echo "🧬 CDRSeqGen: Starting Automated Experiment Suite"
echo "🎯 Target GPU: $GPU_ID"
echo "📦 Suite: $SUITE"
echo "=========================================================="

# --- Suite 1: OAS-derived matrix ---
# Full 3 (loop) x 2 (model) x 2 (light-chain) factorial, sourced from
# processed_paired_data.json (see dataset/preprocess_paired.py).
# Scratch runs first (cheap/fast) so a bug surfaces before burning time on ESM-2.
OAS_CONFIGS=(
    "configs/h1_scratch.json"
    "configs/h1_scratch_lc.json"
    "configs/h2_scratch.json"
    "configs/h2_scratch_lc.json"
    "configs/h3_scratch.json"
    "configs/h3_scratch_lc.json"
    "configs/h1_esm2.json"
    "configs/h1_esm2_lc.json"
    "configs/h2_esm2.json"
    "configs/h2_esm2_lc.json"
    "configs/h3_esm2.json"
    "configs/h3_esm2_lc.json"
)

# --- Suite 2: SAbDab2 antigen-conditioned matrix ---
# All sourced from processed_sabdab2_data.json (see dataset/preprocess_sabdab2.py).
# The `_ag` runs carry light-chain + antigen context; the `_ag_lconly` runs are the matched
# ablation baselines (same SAbDab2 data/split, light chain only, antigen OFF). The valid
# antigen comparison is `_ag` vs `_ag_lconly` on the SAME data -- NOT `_ag` vs the OAS `_lc`
# runs, which are a different train/test distribution entirely.
# Full 3 (loop) x 2 (model) x 2 (antigen on/off) matrix. Scratch first (cheap), and within each
# model the lconly baseline runs before its matched _ag so the ablation pair lands together.
ANTIGEN_CONFIGS=(
    "configs/h1_scratch_ag_lconly.json"
    "configs/h2_scratch_ag_lconly.json"
    "configs/h3_scratch_ag_lconly.json"
    "configs/h1_scratch_ag.json"
    "configs/h2_scratch_ag.json"
    "configs/h3_scratch_ag.json"
    "configs/h1_esm2_ag_lconly.json"
    "configs/h2_esm2_ag_lconly.json"
    "configs/h3_esm2_ag_lconly.json"
    "configs/h1_esm2_ag.json"
    "configs/h2_esm2_ag.json"
    "configs/h3_esm2_ag.json"
)

# --- Suite 3: antibody/protein BERT PLM backbones (OAS) ---
# Same OAS data + 3 (loop) x 2 (light-chain) matrix as Suite 1, but with the protein_bert
# backbone (see backbones.py). IgBert (Exscientia, antibody-specific, OAS-paired-trained) is the
# antibody PLM; ProtBert (Rostlab) is the general-protein baseline. NOTE: the first run of each
# downloads a large (~1.6GB) model from the HF Hub, which is why these are opt-in suites and not
# bundled into "all".
IGBERT_CONFIGS=(
    "configs/h1_igbert.json"
    "configs/h1_igbert_lc.json"
    "configs/h2_igbert.json"
    "configs/h2_igbert_lc.json"
    "configs/h3_igbert.json"
    "configs/h3_igbert_lc.json"
)
PROTBERT_CONFIGS=(
    "configs/h1_protbert.json"
    "configs/h1_protbert_lc.json"
    "configs/h2_protbert.json"
    "configs/h2_protbert_lc.json"
    "configs/h3_protbert.json"
    "configs/h3_protbert_lc.json"
)

CONFIGS=()
case "$SUITE" in
    oas)      CONFIGS=("${OAS_CONFIGS[@]}") ;;
    antigen)  CONFIGS=("${ANTIGEN_CONFIGS[@]}") ;;
    all)      CONFIGS=("${OAS_CONFIGS[@]}" "${ANTIGEN_CONFIGS[@]}") ;;
    igbert)   CONFIGS=("${IGBERT_CONFIGS[@]}") ;;
    protbert) CONFIGS=("${PROTBERT_CONFIGS[@]}") ;;
    plm)      CONFIGS=("${IGBERT_CONFIGS[@]}" "${PROTBERT_CONFIGS[@]}") ;;
    *) echo "❌ Unknown suite '$SUITE'. Use: oas | antigen | all | igbert | protbert | plm."; exit 1 ;;
esac

# Fail fast with a clear message if the required processed data is missing,
# rather than burning GPU time before crashing on the first config that needs it.
# oas, all, and the PLM suites (igbert/protbert/plm) all train on the OAS paired data.
if [[ "$SUITE" =~ ^(oas|all|igbert|protbert|plm)$ ]] \
    && [[ ! -f dataset/data/processed/processed_paired_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_paired_data.json — run dataset/preprocess_paired.py first."
    exit 1
fi
if [[ "$SUITE" == "antigen" || "$SUITE" == "all" ]] \
    && [[ ! -f dataset/data/processed/processed_sabdab2_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_sabdab2_data.json — run:"
    echo "     bash dataset/data/download_sabdab2.sh"
    echo "     python dataset/preprocess_sabdab2.py"
    exit 1
fi

# Export the GPU variable for all Python runs
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Loop through and run each config
for config in "${CONFIGS[@]}"; do
    echo ""
    echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] Starting run: $config"
    echo "----------------------------------------------------------"

    # Run the training script.
    # || true ensures that if a run crashes, the loop continues to the next config.
    python train.py "$config" || { echo "❌ WARNING: Run failed for $config"; true; }

    echo "✅ Finished run: $config"
done

echo ""
echo "🎉 All experiments completed successfully!"
echo "Check the 'results/' directory for your timestamped folders and CSV metrics."
