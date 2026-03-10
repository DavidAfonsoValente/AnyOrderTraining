#!/bin/bash
# scripts/run_single_ablation.sh
# Runs training followed by eval for a specific mask probability.
# Optimized for the fast 'gpu' partition to bypass long-queue delays.
#SBATCH --job-name=aomt_ablation
#SBATCH --output=logs/ablation_%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -C "h100|a100"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu

set -euo pipefail
PROB=$1
P_NAME="p$(echo $PROB | tr '.' '_')"
OUT_DIR="./checkpoints/ablation_${P_NAME}"

mkdir -p "$OUT_DIR" logs/ablation

source scripts/_train_common.sh

# 1. Generate temp config
# We use the native tokenizer path for ablation to ensure no path issues
sed "s/mask_prob: 0.25/mask_prob: $PROB/" configs/aomt_mixed.yaml > "/tmp/config_${P_NAME}.yaml"
sed -i "s|output_dir:.*|output_dir: $OUT_DIR|" "/tmp/config_${P_NAME}.yaml"
sed -i "s|train_path:.*|train_path: ./data/cache/ablation/aomt_alfworld_train.jsonl|" "/tmp/config_${P_NAME}.yaml"
sed -i "s|num_epochs:.*|num_epochs: 3|" "/tmp/config_${P_NAME}.yaml"

# 2. Train
echo "[$(date)] Starting Ablation: mask_prob=$PROB"
launch_training tasks/train_aomt.py "/tmp/config_${P_NAME}.yaml"

# 3. Convert for Eval
echo "[$(date)] Converting checkpoint for eval..."
BEST_CKPT=$(ls -td "$OUT_DIR"/*/ | head -1)
EVAL_MODEL="./weights/ablation_${P_NAME}-sep"
python dFactory/scripts/moe_convertor.py \
    --input-path  "${BEST_CKPT}/hf_ckpt" \
    --output-path "$EVAL_MODEL" \
    --mode split
cp ./weights/LLaDA2.0-mini/modeling_llada2_moe.py "$EVAL_MODEL/"

# 4. Evaluate on ALFWorld
echo "[$(date)] Evaluating on ALFWorld..."
python eval/task_eval.py \
    --model_path "$EVAL_MODEL" \
    --tokenizer  "./weights/LLaDA2.0-mini" \
    --benchmark  alfworld \
    --split      test \
    --n_episodes 30 \
    --output_file "results/ablation_${P_NAME}_alfworld.json"

echo "[$(date)] Ablation $PROB complete."
