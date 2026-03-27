#!/bin/bash
#SBATCH --job-name=aomt_standard_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/standard_sft_%j.out
#SBATCH --error=logs/standard_sft_%j.err

set -euo pipefail
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

RUN_NAME="standard_sft"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting Standard SFT training"

# Using train_standard_sft.py as identified in the tree
torchrun --nproc_per_node=4 \
  aomt/tasks/train_standard_sft.py \
    --model_name_or_path  ./models/llada2-mini-sep \
    --train_data_path     data/cache/sft_standard_train.jsonl \
    --output_dir          "${OUTPUT_DIR}" \
    --learning_rate       2.5e-5 \
    --lr_scheduler_type   cosine \
    --min_lr              2.5e-6 \
    --warmup_steps        50 \
    --weight_decay        0.1 \
    --max_grad_norm       1.0 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs    3 \
    --max_seq_length      2048 \
    --bf16 \
    --gradient_checkpointing \
    --save_strategy       epoch \
    --logging_steps       10 \
    --dataloader_num_workers 4

echo "[$(date)] Standard SFT training complete. Output: ${OUTPUT_DIR}"
