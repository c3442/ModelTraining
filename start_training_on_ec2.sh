#!/bin/bash
set -e

echo "=================================================="
echo "🚀 Starting AWS A100 Training Bootstrap"
echo "=================================================="

# -------------------------------
# Basic environment
# -------------------------------
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# -------------------------------
# Hugging Face Authentication
# Need to set manually $HF_TOKEN on the machine
# -------------------------------
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN   # for compatibility with older libs
export HF_HOME=$HOME/.cache/huggingface
mkdir -p $HF_HOME

# -------------------------------
# Project setup
# -------------------------------
mkdir -p ~/train
cd ~/train

echo "📥 Downloading training files from S3..."

aws s3 cp s3://8up-model-training/script/fine_tune_gemma.py .
aws s3 cp s3://8up-model-training/script/requirements.txt .

# -------------------------------
# Python environment
# -------------------------------
sudo dnf install -y python3.12 python3.12-pip python3.12-devel
sudo dnf groupinstall -y "Development Tools"
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf clean all
sudo dnf install -y cuda-toolkit-12-4

# Set the CUDA Home directory (default install location)
export CUDA_HOME=/usr/local/cuda-12.4
# Add the compiler (nvcc) to your PATH
export PATH=${CUDA_HOME}/bin:${PATH}
# Add the libraries to your Library Path
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

#if [ ! -d "venv" ]; then
#  echo "🐍 Creating virtual environment..."
#  python3.12 -m venv venv312
#fi

echo "🐍 Creating virtual environment..."
python3.12 -m venv venv312
source venv312/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip

# Install build tools
sudo dnf install -y gcc gcc-c++ git make cmake ninja-build kernel-devel
pip install psutil packaging wheel setuptools ninja numpy
pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124
# This installs Flash Attention from source, building it against PyTorch 2.9 + CUDA 13
pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

echo "📦 Installing remaining dependencies with no-build-isolation..."
export FLASH_ATTENTION_FORCE_CUDA_ARCH="8.6"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
pip install -r requirements.txt --no-build-isolation --no-cache-dir

# -------------------------------
# Sanity checks
# -------------------------------
echo "✅ Python version:"
python --version

echo "✅ CUDA check:"
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# -------------------------------
# Launch training
# -------------------------------
echo "🔥 Starting training..."
python fine_tune_gemma.py

echo "✅ Training job finished."
