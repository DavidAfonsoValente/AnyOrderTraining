#!/bin/bash
#
# Master script to run the dFactory experiment phases.
#
# This script automates setup and submission of jobs to the Slurm cluster.
#
# Usage:
#   ./run_project.sh <phase>
#
# Arguments:
#   setup_only  - Only runs the environment setup.
#   phase1      - Submits the data and model preparation job.
#   phase2      - Submits the MiniGrid ablation job array.
#   phase3      - Submits the ALFWorld training and evaluation job arrays.
#   all         - Runs setup, then submits all phases sequentially.
#

set -e

# --- 1. Run Environment Setup ---
# This ensures pyenv, uv, and all dependencies are installed and configured.
echo ">>> Running Environment Setup... <<<"
# We use bash to run it, not source, to avoid polluting the user's current shell
bash setup_cluster_env.sh
# After setup, we still need to source the venv for the sbatch command itself
source dFactory/VeOmni/.venv/bin/activate
echo ">>> Environment is ready. <<<"
echo ""


# --- 2. Phase Execution Logic ---
PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Error: No phase specified."
    echo "Usage: ./run_project.sh [setup_only|phase1|phase2|phase3|all]"
    exit 1
fi

cd dFactory

# Function to run Phase 1
run_phase1() {
    echo ">>> Submitting Phase 1: Data and Model Preparation job... <<<"
    sbatch phase1_prepare_data.slurm
    echo "Phase 1 job submitted."
}

# Function to run Phase 2
run_phase2() {
    echo ">>> Submitting Phase 2: MiniGrid Ablation job array (72 jobs)... <<<"
    sbatch --array=0-71 phase2_minigrid_ablation.slurm
    echo "Phase 2 job array submitted."
}

# Function to run Phase 3
run_phase3() {
    echo ">>> Submitting Phase 3: ALFWorld Training and Evaluation jobs... <<<"
    TRAIN_JOB_ID=$(sbatch --parsable --array=0-8 phase3_alfworld_train_llada.slurm)
    echo "Training job array submitted with ID: $TRAIN_JOB_ID"
    
    sbatch --dependency=afterok:$TRAIN_JOB_ID --array=0-8 phase3_alfworld_eval_llada.slurm
    echo "Evaluation job array submitted, will run after training completes successfully."
}


# --- Main Case Statement ---
case $PHASE in
    setup_only)
        echo "Setup complete. Exiting as requested."
        ;;
    phase1)
        run_phase1
        ;;
    phase2)
        run_phase2
        ;;
    phase3)
        run_phase3
        ;;
    all)
        echo ">>> Running all phases sequentially... <<<"
        # For 'all', we need to wait for each phase to complete.
        # This is a simplified version; true sequential submission would
        # use --dependency chaining for all slurm jobs.
        echo "Submitting Phase 1 and waiting for it to complete..."
        JOB1_ID=$(sbatch --parsable --wait phase1_prepare_data.slurm)
        echo "Phase 1 (Job ID: $JOB1_ID) completed."
        
        echo "Submitting Phase 2..."
        run_phase2

        echo "Submitting Phase 3..."
        run_phase3
        ;;
    *)
        echo "Error: Invalid phase '$PHASE'."
        echo "Usage: ./run_project.sh [setup_only|phase1|phase2|phase3|all]"
        exit 1
        ;;
esac

echo ""
echo "Script finished."

