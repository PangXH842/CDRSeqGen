#!/bin/bash

# Light-chain CDR (L1-L3) infilling suite — the mirror of the heavy-chain OAS matrix in run_all.sh.
# Targets the light chain's cdr1/cdr2/cdr3 (target_region = lcdr1/lcdr2/lcdr3); the optional
# partner-chain context here is the HEAVY chain (include_heavy_chain), analogous to how the heavy
# suite prepends the light chain. All runs source processed_paired_data.json (OAS paired data).

# Default to GPU 0 if the user doesn't provide one
GPU_ID=${1:-0}

# Activate the environment (Adjust the path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj

echo "=========================================================="
echo "🧬 CDRSeqGen: Light-Chain CDR (L1-L3) Infilling Suite"
echo "🎯 Target GPU: $GPU_ID"
echo "=========================================================="

# Full 3 (loop) x 2 (model) x 2 (heavy-chain context on/off) factorial.
# Scratch runs first (cheap/fast) so a bug surfaces before burning time on ESM-2; within each loop
# the no-context base runs before its matched _hc so the ablation pair lands together.
LCDR_CONFIGS=(
    "configs/l1_scratch.json"
    "configs/l1_scratch_hc.json"
    "configs/l2_scratch.json"
    "configs/l2_scratch_hc.json"
    "configs/l3_scratch.json"
    "configs/l3_scratch_hc.json"
    "configs/l1_esm2.json"
    "configs/l1_esm2_hc.json"
    "configs/l2_esm2.json"
    "configs/l2_esm2_hc.json"
    "configs/l3_esm2.json"
    "configs/l3_esm2_hc.json"
)

# Fail fast with a clear message if the required processed data is missing,
# rather than burning GPU time before crashing on the first config.
if [[ ! -f dataset/data/processed/processed_paired_data.json ]]; then
    echo "❌ Missing dataset/data/processed/processed_paired_data.json — run dataset/preprocess_paired.py first."
    exit 1
fi

# Export the GPU variable for all Python runs
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Loop through and run each config
for config in "${LCDR_CONFIGS[@]}"; do
    echo ""
    echo "▶️ [$(date +'%Y-%m-%d %H:%M:%S')] Starting run: $config"
    echo "----------------------------------------------------------"

    # || true ensures that if a run crashes, the loop continues to the next config.
    python train.py "$config" || { echo "❌ WARNING: Run failed for $config"; true; }

    echo "✅ Finished run: $config"
done

echo ""
echo "🎉 All L-CDR experiments completed!"
echo "Check the 'results/' directory for timestamped folders (e.g. *_lcdr3_OAS_HC_scratch) and CSV metrics."
