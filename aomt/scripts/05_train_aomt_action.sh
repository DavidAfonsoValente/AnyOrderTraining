#!/bin/bash
# =============================================================================
# 05_train_aomt_action.sh — AOMT-Action-Only
# Full trajectories (longer sequences). batch=1, grad_accum=16.
# =============================================================================
#SBATCH --job-name=aomt_act_only
#SBATCH --output=logs/05_aomt_action_%j.log
#SBATCH --time=20:00:00           # 5 epochs, longer than baselines
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=h100-96:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --partition=normal

set -euo pipefail
mkdir -p logs checkpoints/aomt_action_only

echo "[$(date)] Starting AOMT-Action-Only | Job: $SLURM_JOB_ID"
source scripts/_train_common.sh

launch_training tasks/train_aomt.py configs/aomt_action_only.yaml

echo "[$(date)] AOMT-Action-Only done."

# ---- Mask probability ablation on ALFWorld ----------------------------------
# Uncomment to run ablations after the primary experiment.
# for P in 0.15 0.40 0.50; do
#     sed "s/mask_prob: 0.25/mask_prob: $P/" \
#         configs/aomt_action_only.yaml > /tmp/aomt_act_p${P}.yaml
#     sed -i "s|output_dir:.*|output_dir: ./checkpoints/aomt_act_p${P}|" \
#         /tmp/aomt_act_p${P}.yaml
#     launch_training tasks/train_aomt.py /tmp/aomt_act_p${P}.yaml
# done
