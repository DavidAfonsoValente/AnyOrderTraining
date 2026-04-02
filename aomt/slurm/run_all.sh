#!/bin/bash
set -euo pipefail

echo "Submitting AOMT Training Jobs..."

# Main training methods
J_SFT=$(sbatch --parsable aomt/slurm/train_standard_sft.sh)
J_PFX1=$(sbatch --parsable aomt/slurm/train_prefix_sft_stage1.sh)
J_AOA=$(sbatch --parsable aomt/slurm/train_aomt_action_only.sh)
J_MIX=$(sbatch --parsable aomt/slurm/train_aomt_mixed_p025.sh)

# Stage 2 depends on Stage 1
J_PFX2=$(sbatch --parsable --dependency=afterok:$J_PFX1 aomt/slurm/train_prefix_sft_stage2.sh)

echo "Jobs submitted:"
echo "  Standard SFT: $J_SFT"
echo "  Prefix SFT S1: $J_PFX1"
echo "  Prefix SFT S2: $J_PFX2 (depends on $J_PFX1)"
echo "  AOMT Action: $J_AOA"
echo "  AOMT Mixed: $J_MIX"

# Evaluation (example - would need eval_all.sh)
# sbatch --dependency=afterok:$J_MIX:$J_PFX2:$J_SFT:$J_AOA aomt/slurm/evaluate_all.sh
