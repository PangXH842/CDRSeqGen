#!/bin/bash

# Default to GPU 0 if the user doesn't provide one
GPU_ID=${1:-0}

# Activate the environment (Adjust the path if your conda init is different)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cdrprj

echo "=========================================================="
echo "🧬 CDRSeqGen: Starting Automated Experiment Suite"
echo "🎯 Target GPU: $GPU_ID"
echo "=========================================================="

# Define the matrix of experiments
# Full 3 (loop) x 2 (model) x 2 (light-chain) factorial.
# Scratch runs first (cheap/fast) so a bug surfaces before burning time on ESM-2.
CONFIGS=(
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