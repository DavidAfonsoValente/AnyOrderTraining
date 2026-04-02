#!/bin/bash
#
# AOMT Evaluation Script
#
# This script is a placeholder for running the full evaluation suite on a
# trained model checkpoint.
#
# A complete implementation would:
# 1. Take a checkpoint path and a dataset split (e.g., test_seen, test_unseen) as arguments.
# 2. Load the model and tokenizer from the checkpoint.
# 3. Load and process the specified dataset split.
# 4. Run `eval/nll_obs.py` to get the NLL-obs score.
# 5. Run `eval/nll_act.py` to get the NLL-act score.
# 6. Run `eval/task_eval.py` on all relevant environment tasks (ALFWorld, ScienceWorld, etc.).
# 7. Run `eval/noise_robustness.py` to get robustness scores.
# 8. Aggregate all results into a final JSON or CSV report.
#
# Slurm Usage:
# #!/bin/sh
# #SBATCH --job-name=aomt_eval
# #SBATCH --output=slurm_logs/aomt_eval_%j.out
# #SBATCH --gpus=a100:2
# #SBATCH --time=8:00:00
#
# srun python3 -u run_full_eval.py --checkpoint_path "$1" --split "$2"
#

set -e

# --- Placeholder Arguments ---
CHECKPOINT_PATH=$1
DATA_SPLIT=$2

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$DATA_SPLIT" ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <path_to_checkpoint> <data_split>"
    echo "Example: $0 ./checkpoints/aomt_mixed/final_model test_unseen"
    exit 1
fi

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT="$SCRIPT_DIR/.."

echo "========================================"
echo "          AOMT Evaluation Run"
echo "========================================"
echo "Checkpoint:   $CHECKPOINT_PATH"
echo "Data Split:   $DATA_SPLIT"
echo "----------------------------------------"
echo "NOTE: This is a demonstration script."
echo "It will run the example code from the evaluation modules."

# In a real implementation, you would have a master script `run_full_eval.py`
# that orchestrates these calls. For now, we call the main blocks of each
# eval script to show they are functional.

echo "\n--- Running NLL-Obs Demonstration ---"
# This would be pointed at the specified data split
python3 "$PROJECT_ROOT/eval/nll_obs.py"

echo "\n--- Running NLL-Act Demonstration ---"
python3 "$PROJECT_ROOT/eval/nll_act.py"

echo "\n--- Running Task Eval Demonstration ---"
python3 "$PROJECT_ROOT/eval/task_eval.py"

echo "\n--- Running Noise Robustness Demonstration ---"
python3 "$PROJECT_ROOT/eval/noise_robustness.py"


echo "----------------------------------------"
echo "Evaluation run finished."
echo "========================================"
