#!/bin/bash
set -e

echo "=================================================="
echo "🧪 Starting AWS A100 Evaluation Bootstrap"
echo "=================================================="

# -------------------------------
# Basic environment
# -------------------------------
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# -------------------------------
# Hugging Face Authentication
# -------------------------------
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN   # for compatibility with older libs
export HF_HOME=$HOME/.cache/huggingface
mkdir -p "$HF_HOME"

# -------------------------------
# Project setup
# -------------------------------
mkdir -p ~/eval
cd ~/eval

echo "📥 Downloading evaluation files from S3..."
aws s3 cp s3://8up-model-training/script/eval_gemma.py .
aws s3 cp s3://8up-model-training/script/requirements.txt .

# Optional: pull a fine-tuned adapter from S3 if provided
if [ -n "$ADAPTER_S3" ]; then
  echo "📦 Syncing adapter from $ADAPTER_S3 ..."
  aws s3 sync "$ADAPTER_S3" "${ADAPTER_DIR:-gemma3-nutrition5k-vision-qlora}"
fi

# -------------------------------
# Python environment
# -------------------------------
#if [ ! -d "venv" ]; then
#  echo "🐍 Creating virtual environment..."
#  python3 -m venv venv
#fi
#
#source venv/bin/activate
#
#echo "📦 Installing dependencies..."
#pip install --upgrade pip
#
## Install build tools
#sudo dnf install -y gcc gcc-c++ git make cmake ninja-build python3-devel kernel-devel
#pip install psutil packaging wheel setuptools ninja numpy
#pip install "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124
#pip install flash-attn==2.8.3
#
#echo "📦 Installing remaining dependencies with no-build-isolation..."
#export FLASH_ATTENTION_FORCE_CUDA_ARCH="8.6"
#export CUDA_HOME=/usr/local/cuda-12.9
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#pip install -r requirements.txt --no-build-isolation --no-cache-dir
source ~/train/venv/bin/activate

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
# Launch evaluation
# -------------------------------
EVAL_JSONL=${EVAL_JSONL:-"s3://8up-model-training/training_nutrition5k/test.jsonl"}
IMAGE_BASE=${IMAGE_BASE:-"s3://8up-model-training/images_nutrition5k"}
# ADAPTER_DIR=${ADAPTER_DIR:-"gemma3-nutrition5k-vision-qlora"}   # leave empty for baseline
ADAPTER_DIR=""
OUTPUT_DIR=${OUTPUT_DIR:-"eval-results"}
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
  MAX_SAMPLES_ARG="--max_samples $MAX_SAMPLES"
fi

echo "🧪 Running evaluation..."
python eval_gemma.py \
  --eval_jsonl "$EVAL_JSONL" \
  --image_base_dir "$IMAGE_BASE" \
  --adapter_dir "$ADAPTER_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --load_in_4bit \
  $MAX_SAMPLES_ARG

echo "✅ Evaluation job finished."
