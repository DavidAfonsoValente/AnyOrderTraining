#!/bin/bash
#SBATCH --job-name=aomt_steps_ablation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_steps_ablation_%j.out
#SBATCH --error=logs/eval_steps_ablation_%j.err

set -euo pipefail
export PYTHONPATH=.

RESULTS_DIR="eval/results/steps_ablation"
mkdir -p "${RESULTS_DIR}" logs

GEN_LENGTH=256
CHECKPOINTS_standard="outputs/standard_sft/epoch_2"
CHECKPOINTS_aomt="outputs/aomt_mixed_final/epoch_4"

STEPS_VALUES=(1 8 32 128)

for METHOD in standard_sft aomt_mixed; do
    if [ "$METHOD" == "standard_sft" ]; then CKPT=$CHECKPOINTS_standard; else CKPT=$CHECKPOINTS_aomt; fi
    if [ ! -d "${CKPT}" ]; then continue; fi
    for STEPS in "${STEPS_VALUES[@]}"; do
        OUT="${RESULTS_DIR}/${METHOD}_steps${STEPS}_alfworld_seen.json"
        echo "[$(date)] Eval: method=${METHOD} steps=${STEPS}"
        python3 eval/task_eval.py \
            --method         "${METHOD}" \
            --checkpoint_dir "${CKPT}" \
            --benchmark      alfworld \
            --split          eval_in_distribution \
            --steps          "${STEPS}" \
            --gen_length     "${GEN_LENGTH}" \
            --output_json    "${OUT}" \
            2>&1 | tee "${RESULTS_DIR}/${METHOD}_steps${STEPS}.log"
    done
done

echo "[$(date)] Steps ablation complete"
