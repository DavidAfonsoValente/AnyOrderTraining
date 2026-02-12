#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --time=120
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dvalente@comp.nus.edu.sg

source venv/bin/activate

echo "=========================================="
echo "Evaluating All Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# Find all experiment directories
exp_dirs=(outputs/ablations/exp*)

if [ ${#exp_dirs[@]} -eq 0 ]; then
    echo "ERROR: No experiment directories found!"
    exit 1
fi

echo "Found ${#exp_dirs[@]} experiment directories"
echo ""

# Evaluate each experiment
for exp_dir in "${exp_dirs[@]}"; do
    if [ -d "$exp_dir" ] && [ -d "$exp_dir/checkpoints" ]; then
        echo "=========================================="
        echo "Evaluating: $exp_dir"
        echo "=========================================="
        
        # Check if best checkpoint exists
        if [ -f "$exp_dir/checkpoints/best.pt" ]; then
            python scripts/evaluate.py \
                --checkpoint $exp_dir/checkpoints/best.pt \
                --metric all \
                --data_dir data/raw/BabyAI-GoToRedBall-v0 \
                --env BabyAI-GoToRedBall-v0 \
                --output $exp_dir/results.json
            
            echo "✓ Evaluation complete for $exp_dir"
            echo ""
        else
            echo "⚠ No best checkpoint found in $exp_dir"
            echo ""
        fi
    fi
done

echo "=========================================="
echo "All Evaluations Complete!"
echo "Finished: $(date)"
echo "=========================================="

# Generate summary
echo ""
echo "Results Summary:"
echo "----------------------------------------"
for exp_dir in "${exp_dirs[@]}"; do
    if [ -f "$exp_dir/results.json" ]; then
        echo "$exp_dir:"
        python3 -c "
import json
import sys
try:
    with open('$exp_dir/results.json', 'r') as f:
        data = json.load(f)
    print(f\"  NLL: {data.get('world_model_nll', 'N/A'):.4f}\" if isinstance(data.get('world_model_nll'), (int, float)) else '  NLL: N/A')
    print(f\"  Obs Acc: {data.get('observation_accuracy', 'N/A'):.4f}\" if isinstance(data.get('observation_accuracy'), (int, float)) else '  Obs Acc: N/A')
    print(f\"  Act Acc: {data.get('action_accuracy', 'N/A'):.4f}\" if isinstance(data.get('action_accuracy'), (int, float)) else '  Act Acc: N/A')
except Exception as e:
    print(f'  Error reading results: {e}')
"
        echo ""
    fi
done
