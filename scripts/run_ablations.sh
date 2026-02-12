#!/bin/bash
# Automated Ablation Study Script
# Runs the complete set of ablation experiments for Phase 1

set -e

echo "================================================"
echo "Any-Order Masked Training - Ablation Study"
echo "================================================"

# Configuration
DATA_DIR="data/raw"
OUTPUT_BASE="outputs/ablations"
NUM_EPOCHS=50
BATCH_SIZE=8

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +%H:%M:%S)]${NC} $1"
}

# Create output directory
mkdir -p $OUTPUT_BASE

# ================================================
# Experiment 1: Masking Probability Ablation
# ================================================
print_status "Starting Experiment 1: Masking Probability Ablation"

for mask_prob in 0.15 0.30 0.50; do
    exp_name="exp1_cell_p$(echo $mask_prob | tr -d '.')"
    output_dir="$OUTPUT_BASE/$exp_name"
    
    print_status "Training with mask_prob=$mask_prob..."
    
    python scripts/train.py \
        --config configs/experiments/cell_masking.yaml \
        --masking.mask_prob $mask_prob \
        --training.num_epochs $NUM_EPOCHS \
        --data.batch_size $BATCH_SIZE \
        --data.data_dir $DATA_DIR \
        --output_dir $output_dir \
        2>&1 | tee $output_dir/training.log
    
    print_status "Evaluating..."
    python scripts/evaluate.py \
        --checkpoint $output_dir/checkpoints/best.pt \
        --metric all \
        --data_dir $DATA_DIR/BabyAI-GoToRedBall-v0 \
        --env BabyAI-GoToRedBall-v0 \
        --output $output_dir/results.json
    
    print_status "Completed mask_prob=$mask_prob"
    echo ""
done

# ================================================
# Experiment 2: Cell vs Attribute Masking
# ================================================
print_status "Starting Experiment 2: Cell vs Attribute Masking"

# Cell-level
exp_name="exp2_cell"
output_dir="$OUTPUT_BASE/$exp_name"

print_status "Training cell-level masking..."
python scripts/train.py \
    --config configs/experiments/cell_masking.yaml \
    --training.num_epochs $NUM_EPOCHS \
    --data.batch_size $BATCH_SIZE \
    --data.data_dir $DATA_DIR \
    --output_dir $output_dir \
    2>&1 | tee $output_dir/training.log

print_status "Evaluating cell-level..."
python scripts/evaluate.py \
    --checkpoint $output_dir/checkpoints/best.pt \
    --metric all \
    --data_dir $DATA_DIR/BabyAI-GoToRedBall-v0 \
    --env BabyAI-GoToRedBall-v0 \
    --output $output_dir/results.json

# Attribute-level
exp_name="exp2_attribute"
output_dir="$OUTPUT_BASE/$exp_name"

print_status "Training attribute-level masking..."
python scripts/train.py \
    --config configs/experiments/attribute_masking.yaml \
    --training.num_epochs $NUM_EPOCHS \
    --data.batch_size $BATCH_SIZE \
    --data.data_dir $DATA_DIR \
    --output_dir $output_dir \
    2>&1 | tee $output_dir/training.log

print_status "Evaluating attribute-level..."
python scripts/evaluate.py \
    --checkpoint $output_dir/checkpoints/best.pt \
    --metric all \
    --data_dir $DATA_DIR/BabyAI-GoToRedBall-v0 \
    --env BabyAI-GoToRedBall-v0 \
    --output $output_dir/results.json

print_status "Completed Experiment 2"
echo ""

# ================================================
# Experiment 3: Scheduled vs Fixed Masking
# ================================================
print_status "Starting Experiment 3: Scheduled vs Fixed Masking"

# Scheduled
exp_name="exp3_scheduled"
output_dir="$OUTPUT_BASE/$exp_name"

print_status "Training with scheduled masking..."
python scripts/train.py \
    --config configs/experiments/scheduled_masking.yaml \
    --training.num_epochs $NUM_EPOCHS \
    --data.batch_size $BATCH_SIZE \
    --data.data_dir $DATA_DIR \
    --output_dir $output_dir \
    2>&1 | tee $output_dir/training.log

print_status "Evaluating scheduled..."
python scripts/evaluate.py \
    --checkpoint $output_dir/checkpoints/best.pt \
    --metric all \
    --data_dir $DATA_DIR/BabyAI-GoToRedBall-v0 \
    --env BabyAI-GoToRedBall-v0 \
    --output $output_dir/results.json

# Fixed (reuse from Experiment 2)
print_status "Fixed baseline results already available from Experiment 2"
echo ""

# ================================================
# Generate Comparison Report
# ================================================
print_status "Generating comparison report..."

cat > $OUTPUT_BASE/RESULTS_SUMMARY.md << EOF
# Ablation Study Results Summary

## Experiment 1: Masking Probability

| Mask Prob | World Model NLL | Obs Accuracy | Action Accuracy | Success Rate |
|-----------|-----------------|--------------|-----------------|--------------|
EOF

for mask_prob in 0.15 0.30 0.50; do
    exp_name="exp1_cell_p$(echo $mask_prob | tr -d '.')"
    results_file="$OUTPUT_BASE/$exp_name/results.json"
    
    if [ -f "$results_file" ]; then
        nll=$(python -c "import json; print(json.load(open('$results_file'))['world_model_nll'])" 2>/dev/null || echo "N/A")
        obs_acc=$(python -c "import json; print(json.load(open('$results_file'))['observation_accuracy'])" 2>/dev/null || echo "N/A")
        act_acc=$(python -c "import json; print(json.load(open('$results_file'))['action_accuracy'])" 2>/dev/null || echo "N/A")
        success=$(python -c "import json; print(json.load(open('$results_file'))['task_success']['success_rate'])" 2>/dev/null || echo "N/A")
        
        echo "| $mask_prob | $nll | $obs_acc | $act_acc | $success |" >> $OUTPUT_BASE/RESULTS_SUMMARY.md
    fi
done

cat >> $OUTPUT_BASE/RESULTS_SUMMARY.md << EOF

## Experiment 2: Masking Strategy

| Strategy | World Model NLL | Obs Accuracy | Action Accuracy | Success Rate |
|----------|-----------------|--------------|-----------------|--------------|
EOF

for strategy in cell attribute; do
    exp_name="exp2_$strategy"
    results_file="$OUTPUT_BASE/$exp_name/results.json"
    
    if [ -f "$results_file" ]; then
        nll=$(python -c "import json; print(json.load(open('$results_file'))['world_model_nll'])" 2>/dev/null || echo "N/A")
        obs_acc=$(python -c "import json; print(json.load(open('$results_file'))['observation_accuracy'])" 2>/dev/null || echo "N/A")
        act_acc=$(python -c "import json; print(json.load(open('$results_file'))['action_accuracy'])" 2>/dev/null || echo "N/A")
        success=$(python -c "import json; print(json.load(open('$results_file'))['task_success']['success_rate'])" 2>/dev/null || echo "N/A")
        
        echo "| $strategy | $nll | $obs_acc | $act_acc | $success |" >> $OUTPUT_BASE/RESULTS_SUMMARY.md
    fi
done

cat >> $OUTPUT_BASE/RESULTS_SUMMARY.md << EOF

## Experiment 3: Scheduled vs Fixed Masking

| Type | World Model NLL | Obs Accuracy | Action Accuracy | Success Rate |
|------|-----------------|--------------|-----------------|--------------|
EOF

for type in scheduled cell; do
    if [ "$type" = "scheduled" ]; then
        exp_name="exp3_scheduled"
    else
        exp_name="exp2_cell"
    fi
    results_file="$OUTPUT_BASE/$exp_name/results.json"
    
    if [ -f "$results_file" ]; then
        nll=$(python -c "import json; print(json.load(open('$results_file'))['world_model_nll'])" 2>/dev/null || echo "N/A")
        obs_acc=$(python -c "import json; print(json.load(open('$results_file'))['observation_accuracy'])" 2>/dev/null || echo "N/A")
        act_acc=$(python -c "import json; print(json.load(open('$results_file'))['action_accuracy'])" 2>/dev/null || echo "N/A")
        success=$(python -c "import json; print(json.load(open('$results_file'))['task_success']['success_rate'])" 2>/dev/null || echo "N/A")
        
        echo "| $type | $nll | $obs_acc | $act_acc | $success |" >> $OUTPUT_BASE/RESULTS_SUMMARY.md
    fi
done

cat >> $OUTPUT_BASE/RESULTS_SUMMARY.md << EOF

## Notes

- All experiments run on BabyAI-GoToRedBall-v0
- Training: $NUM_EPOCHS epochs, batch size $BATCH_SIZE
- Generated on: $(date)
EOF

print_status "Results summary saved to $OUTPUT_BASE/RESULTS_SUMMARY.md"

# ================================================
# Complete
# ================================================
echo ""
print_status "================================================"
print_status "Ablation Study Complete!"
print_status "================================================"
print_status "Results saved to: $OUTPUT_BASE/"
print_status "Summary report: $OUTPUT_BASE/RESULTS_SUMMARY.md"
print_status ""
print_status "To view individual experiment logs:"
print_status "  ls $OUTPUT_BASE/exp*/training.log"
print_status ""
print_status "To compare experiments with tensorboard:"
print_status "  tensorboard --logdir $OUTPUT_BASE"
