#!/bin/bash
# scripts/fix_environment.sh
set -e

PROJECT_DIR="/home/d/dvalente/AnyOrderTraining"
cd "$PROJECT_DIR"

source venv/bin/activate

echo "Installing Training Optimizers (PEFT, BitsAndBytes, Accelerate)..."
export PIP_CACHE_DIR="/tmp/$USER/pip_cache"
mkdir -p "$PIP_CACHE_DIR"

# Install all necessary libraries for QLoRA 16B training
pip install --no-cache-dir \
    peft \
    bitsandbytes \
    accelerate \
    scipy \
    sentencepiece

echo "Verifying all modules..."
python3 -c "import torch; import peft; import bitsandbytes; print(f'SUCCESS: CUDA={torch.cuda.is_available()}, PEFT and BNB loaded.')"
