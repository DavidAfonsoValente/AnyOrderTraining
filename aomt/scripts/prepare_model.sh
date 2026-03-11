#!/bin/bash
# =============================================================================
# prepare_model.sh — Model Download and MoE Expert Merging
#
# Hardware: Required high VRAM (A100-80 or H100) for merging.
# =============================================================================
#SBATCH --job-name=aomt_prep_model
#SBATCH --output=logs/prepare_model_%j.log
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu

set -e
mkdir -p logs

echo "========================================"
echo "      AOMT Model Preparation"
echo "========================================"

# --- Configuration ---
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
AOMT_DIR=$(dirname "$SCRIPT_DIR")
VENV_PYTHON="$AOMT_DIR/dFactory/VeOmni/.venv/bin/python"

MODELS_DIR="$AOMT_DIR/weights"
REPO_ID="inclusionAI/LLaDA2.0-mini"
ORIGINAL_MODEL_PATH="$MODELS_DIR/LLaDA2.0-mini"
MERGED_MODEL_PATH="$MODELS_DIR/llada2-mini-merged"
DOWNLOAD_SCRIPT="$AOMT_DIR/dFactory/scripts/download_hf_model.py"
CONVERTOR_SCRIPT="$AOMT_DIR/dFactory/scripts/moe_convertor.py"

# --- Pre-flight Checks ---
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found at $VENV_PYTHON."
    echo "Please run the './setup.sh' script from the 'aomt' directory first."
    exit 1
fi

# --- Download Model if Missing ---
# Check if any .safetensors files exist in the target directory
if [ ! -d "$ORIGINAL_MODEL_PATH" ] || [ -z "$(ls -A "$ORIGINAL_MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
    echo "Original model weights missing or incomplete at $ORIGINAL_MODEL_PATH."
    echo "Downloading from HuggingFace ($REPO_ID)..."
    
    mkdir -p "$MODELS_DIR"
    
    # download_hf_model.py appends the repo name to local_dir
    "$VENV_PYTHON" "$DOWNLOAD_SCRIPT" \
        --repo_id "$REPO_ID" \
        --local_dir "$MODELS_DIR"
    
    echo "Download complete."
else
    echo "Original model weights found at $ORIGINAL_MODEL_PATH."
fi

if [ ! -f "$CONVERTOR_SCRIPT" ]; then
    echo "Error: MoE convertor script not found at $CONVERTOR_SCRIPT."
    echo "Please ensure the 'dFactory' submodule is cloned correctly."
    exit 1
fi

# --- Main Logic ---
if [ -d "$MERGED_MODEL_PATH" ] && [ -n "$(ls -A "$MERGED_MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
    echo "Merged model weights already exist at '$MERGED_MODEL_PATH'. Skipping conversion."
else
    # If directory exists but is empty/incomplete, remove it to start clean
    if [ -d "$MERGED_MODEL_PATH" ]; then
        echo "Merged model directory exists but appears incomplete. Re-converting..."
        rm -rf "$MERGED_MODEL_PATH"
    fi
    echo "Converting model to 'merged-expert' format for dFactory..."
    echo "Source:      $ORIGINAL_MODEL_PATH"
    echo "Destination: $MERGED_MODEL_PATH"
    
    "$VENV_PYTHON" "$CONVERTOR_SCRIPT" \
        --input-path "$ORIGINAL_MODEL_PATH" \
        --output-path "$MERGED_MODEL_PATH" \
        --mode merge
    
    echo "Conversion complete."
fi

# --- Post-conversion Fix: Copy missing code files ---
# LLaDA uses trust_remote_code=True, so it needs the .py and .json files in the merged directory.
echo "Ensuring all required code and config files are in the merged directory..."
# IMPORTANT: Use the dFactory version of modeling code which supports the merged experts format
cp "$AOMT_DIR/dFactory/models/llada2_moe/modeling_llada2_moe.py" "$MERGED_MODEL_PATH"/
cp "$AOMT_DIR/dFactory/models/llada2_moe/configuration_llada2_moe.py" "$MERGED_MODEL_PATH"/
cp "$AOMT_DIR/dFactory/models/llada2_moe/parallel_plan.py" "$MERGED_MODEL_PATH"/

cp "$ORIGINAL_MODEL_PATH"/*.json "$MERGED_MODEL_PATH"/ 2>/dev/null || true
# Tokenizer files are also needed
cp "$ORIGINAL_MODEL_PATH"/*.model "$MERGED_MODEL_PATH"/ 2>/dev/null || true

# --- Fix model_type for dFactory compatibility ---
# dFactory's modeling_llada2_moe.py uses "llada2_moe_veomni" to signal merged weights
echo "Patching config.json for dFactory compatibility..."
sed -i 's/"model_type": "llada2_moe"/"model_type": "llada2_moe_veomni"/' "$MERGED_MODEL_PATH/config.json"

echo -e "\nModel preparation complete. Use the path '$MERGED_MODEL_PATH' in your training configs."
echo "========================================"
