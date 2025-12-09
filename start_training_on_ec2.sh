#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Starting AWS Training Bootstrap"
echo "=================================================="

if [ -z "$HF_TOKEN" ]; then
  echo "❌ ERROR: HF_TOKEN environment variable is not set."
  echo "   Please run: export HF_TOKEN=your_token_here"
  echo "   before running this script."
  exit 1
fi
echo "✅ HF_TOKEN is set."

# -------------------------------
# Environment Setup
# -------------------------------
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME
mkdir -p ~/train
cd ~/train

# -------------------------------
# System Dependencies (Skipped if already installed)
# -------------------------------
# Checking for nvcc to see if we need to install system packages
if ! command -v nvcc &> /dev/null; then
    echo "⚙️ Installing System Dependencies and CUDA..."
    sudo dnf install -y python3.12 python3.12-pip python3.12-devel git make cmake ninja-build gcc gcc-c++
    sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
    sudo dnf clean all
    sudo dnf install -y cuda-toolkit-12-4
else
    echo "✅ CUDA Toolkit found. Skipping system install."
fi

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# -------------------------------
# Python Environment
# -------------------------------
if [ ! -d "venv312" ]; then
    echo "🐍 Creating new virtual environment..."
    python3.12 -m venv venv312
fi

source venv312/bin/activate
pip install --upgrade pip

# -------------------------------
# Smart Installation Logic
# -------------------------------

echo "📥 Syncing scripts..."
aws s3 cp s3://8up-model-training/script/fine_tune_gemma.py .
aws s3 cp s3://8up-model-training/script/requirements.txt .

echo "📦 Updating standard requirements..."
# We install standard reqs every time (it's fast if nothing changed)
pip install psutil packaging wheel setuptools ninja numpy
pip install -r requirements.txt
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124

# CHECK IF FLASH ATTENTION IS ALREADY INSTALLED
if python -c "import flash_attn" 2>/dev/null; then
    echo "✅ Flash Attention is already installed and working. Skipping build."
else
    echo "⚠️ Flash Attention NOT found. Compiling from source (This takes ~10 mins)..."

    export FLASH_ATTENTION_FORCE_CUDA_ARCH="8.0;8.6"

    # Remove the --no-cache-dir flag so if it builds once, it saves the wheel locally for next time
    pip install flash-attn==2.8.3 --no-build-isolation
fi

# -------------------------------
# Launch Training
# -------------------------------
echo "🔥 Starting training..."
python fine_tune_gemma.py

echo "✅ Training job finished."