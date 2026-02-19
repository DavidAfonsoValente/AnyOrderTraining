#!/bin/bash

# This script is used to launch the full suite of Phase 2 ablation studies.
# Each experiment is a separate sbatch command.
#
# By default, all experiments are commented out.
# Uncomment the lines for the experiments you wish to run.
#
# It is recommended to run one task group (e.g., all 'goto' tasks) at a time.
#
# USAGE:
# 1. Edit the SBATCH directives below (e.g., memory, time) if needed.
# 2. Uncomment the 'sbatch' lines for the experiments you want to run.
# 3. Run the script: ./any_order_training/run_ablations.sh

# --- Master SBATCH Configuration ---
# These settings will be inherited by each job.
# You can override them for specific jobs if needed.
SBATCH_DEFAULTS="
--partition=gpu 
--time=04:00:00 
--mem=64G 
--gpus=1 
--ntasks=1 
--cpus-per-task=4 
"

# --- Main Training Script ---
TRAIN_SCRIPT="any_order_training/tasks/train_any_order.py"

# ===================================================
#  TASK: GoToDoor
# ===================================================
echo "--- Submitting GoToDoor Experiments ---"

# --- Baselines ---
# sbatch $SBATCH_DEFAULTS --job-name=goto-causal --output=goto-causal.%j.out --error=goto-causal.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/causal.yaml
# sbatch $SBATCH_DEFAULTS --job-name=goto-prefix --output=goto-prefix.%j.out --error=goto-prefix.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/prefix.yaml

# --- Ablation: Mask Probability ---
# sbatch $SBATCH_DEFAULTS --job-name=goto-p15 --output=goto-p15.%j.out --error=goto-p15.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p15_all.yaml
# sbatch $SBATCH_DEFAULTS --job-name=goto-p30 --output=goto-p30.%j.out --error=goto-p30.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p30_all.yaml
# sbatch $SBATCH_DEFAULTS --job-name=goto-p50 --output=goto-p50.%j.out --error=goto-p50.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p50_all.yaml
# sbatch $SBATCH_DEFAULTS --job-name=goto-p75 --output=goto-p75.%j.out --error=goto-p75.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p75_all.yaml

# --- Ablation: Masking Strategy ---
# sbatch $SBATCH_DEFAULTS --job-name=goto-p50-obs --output=goto-p50-obs.%j.out --error=goto-p50-obs.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p50_obs.yaml
# sbatch $SBATCH_DEFAULTS --job-name=goto-p50-act --output=goto-p50-act.%j.out --error=goto-p50-act.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/goto/any_order_p50_act.yaml


# ===================================================
#  TASK: PickupDist
# ===================================================
echo "--- Submitting PickupDist Experiments ---"

# --- Baselines ---
# sbatch $SBATCH_DEFAULTS --job-name=pickup-causal --output=pickup-causal.%j.out --error=pickup-causal.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/pickup/causal.yaml
# sbatch $SBATCH_DEFAULTS --job-name=pickup-prefix --output=pickup-prefix.%j.out --error=pickup-prefix.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/pickup/prefix.yaml

# ... (and so on for the other ablations)


# ===================================================
#  TASK: Unlock
# ===================================================
echo "--- Submitting Unlock Experiments ---"

# --- Baselines ---
# sbatch $SBATCH_DEFAULTS --job-name=unlock-causal --output=unlock-causal.%j.out --error=unlock-causal.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/unlock/causal.yaml
# sbatch $SBATCH_DEFAULTS --job-name=unlock-prefix --output=unlock-prefix.%j.out --error=unlock-prefix.%j.err $TRAIN_SCRIPT any_order_training/configs/phase2/unlock/prefix.yaml

# ... (and so on for the other ablations)


echo "
Script finished. Check 'squeue -u $USER' to see your submitted jobs."
