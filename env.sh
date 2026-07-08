#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

ENV_NAME="cdrprj"
PYTHON_VERSION="3.11"

echo "🧬 Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Initialize conda for the script session
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "🔥 Installing PyTorch with CUDA support..."
# Note: Using CUDA 12.1 as it is the standard for modern GPUs (RTX 3090, 4090, A100)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "🤗 Installing Hugging Face and Data Science libraries..."
# We explicitly require transformers >= 4.41.0 due to the `eval_strategy` update
# accelerate is required by HuggingFace Trainer
# sentencepiece is required to load the protein-BERT tokenizers (IgBert, ProtBert backbones)
pip install "transformers>=4.41.0" accelerate pandas scikit-learn tqdm sentencepiece

echo "✅ Environment '$ENV_NAME' successfully created!"
echo "👉 To start working, run: conda activate $ENV_NAME"