#!/bin/bash
# =============================================================================
# 01_prepare_data.sh — Data download, length analysis, JSONL generation
# CPU-only job. Run once before any training.
# =============================================================================
#SBATCH --job-name=aomt_data_prep
#SBATCH --output=logs/01_prepare_data_%j.log
#SBATCH --time=60              # 60 minutes should be ample
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal     # adjust if your cluster has a CPU-only partition

set -euo pipefail
mkdir -p logs data/cache

REPO_ROOT="$(pwd)"
# The scripts are now inside aomt/, so REPO_ROOT is aomt/. 
# We need the parent of aomt/ in PYTHONPATH to support 'import aomt...'
PARENT_DIR="$(dirname "$REPO_ROOT")"
export PYTHONPATH="$REPO_ROOT/dFactory/VeOmni:$PARENT_DIR:${PYTHONPATH:-}"

echo "[$(date)] Starting data preparation..."
echo "Node: $(hostname)"

# --- 1. Length analysis (REQUIRED before training) -----------------------
echo "[$(date)] Running length analysis..."
# This script will download agent-eto/eto-sft-trajectory automatically via HF datasets
python data/measure_lengths.py \
    --tokenizer   ./weights/LLaDA2.0-mini \
    --gen_length  256 \
    --max_seq_len 2048 \
    | tee logs/length_analysis.txt

# --- 2. Generate JSONL files per training mode ---------------------------
echo "[$(date)] Generating training JSONL files..."
# This script also downloads/loads the dataset automatically
python data/prepare_data.py \
    --output_dir ./data/cache/ \
    --tokenizer  ./weights/LLaDA2.0-mini

echo ""
echo "=== Generated files ==="
ls -lh data/cache/
echo ""
echo "=== Line counts (datapoints per split) ==="
for f in data/cache/*.jsonl; do
    echo "  $(wc -l < $f) $f"
done

echo "[$(date)] Data preparation complete."
