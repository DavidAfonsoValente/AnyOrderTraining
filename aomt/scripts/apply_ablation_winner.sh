#!/bin/bash
# scripts/apply_ablation_winner.sh
#SBATCH --job-name=aomt_update_config
#SBATCH --output=logs/ablation_update_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10
#SBATCH --partition=normal

set -euo pipefail

echo "[$(date)] Analyzing ablation results..."
BEST_PROB=$(python3 scripts/select_best_mask_prob.py)

echo "[$(date)] WINNER FOUND: mask_prob = $BEST_PROB"

echo "[$(date)] Updating Phase 2 configs..."
# Update both AOMT configs with the winning probability
sed -i "s/mask_prob: .*/mask_prob: $BEST_PROB/" configs/aomt_action_only.yaml
sed -i "s/mask_prob: .*/mask_prob: $BEST_PROB/" configs/aomt_mixed.yaml

echo "[$(date)] Configs updated. Phase 2 will now use optimal hyperparameter."
