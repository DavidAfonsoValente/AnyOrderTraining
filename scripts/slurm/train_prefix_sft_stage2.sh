#!/bin/bash
#SBATCH --job-name=aomt_prefix_stage2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/prefix_sft_stage2_%j.out
#SBATCH --error=logs/prefix_sft_stage2_%j.err

set -euo pipefail
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false

STAGE1_CKPT="${STAGE1_CKPT_PATH:-outputs/prefix_sft_stage1/epoch_2}"
RUN_NAME="prefix_sft_stage2"
OUTPUT_DIR="outputs/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[$(date)] Starting Prefix SFT Stage 2 from: ${STAGE1_CKPT} on 1x A100"

torchrun --nproc_per_node=1 \
  aomt/tasks/train_standard_sft.py \
    --model_name_or_path  "${STAGE1_CKPT}" \
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
    --logging_steps       10

echo "[$(date)] Prefix SFT Stage 2 complete. Output: ${OUTPUT_DIR}"
