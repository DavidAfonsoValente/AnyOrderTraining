#!/bin/bash
# Quick eval: 5 ALFWorld episodes per model, 4 models in PARALLEL on 4 GPUs
set -euo pipefail

export MODELING_BACKEND=hf
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=1
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining:/scratch/e1507650/AnyOrderTraining/aomt/dFactory:${PYTHONPATH:-}"

cd /scratch/e1507650/AnyOrderTraining/aomt
mkdir -p results/quick_eval

TOKENIZER=./weights/LLaDA2.0-mini
CONFIG=./weights/llada2-mini-merged
N=5  # episodes per model

echo "Starting parallel eval on 4 GPUs @ $(date)"

# GPU 0: Standard SFT
CUDA_VISIBLE_DEVICES=0 python3 eval/task_eval.py \
    --model_path ./checkpoints/sft_standard/epoch_2 \
    --config_path $CONFIG --tokenizer $TOKENIZER \
    --benchmark alfworld --n_episodes $N \
    --output_file results/quick_eval/sft_standard_alfworld.json \
    2>&1 | tee results/quick_eval/sft_standard.log &
PID1=$!

# GPU 2: AOMT-Action-Only
CUDA_VISIBLE_DEVICES=2 python3 eval/task_eval.py \
    --model_path ./checkpoints/aomt_action_only/epoch_4 \
    --config_path $CONFIG --tokenizer $TOKENIZER \
    --benchmark alfworld --n_episodes $N \
    --output_file results/quick_eval/aomt_action_only_alfworld.json \
    2>&1 | tee results/quick_eval/aomt_action_only.log &
PID2=$!

# GPU 6: AOMT-Mixed
CUDA_VISIBLE_DEVICES=6 python3 eval/task_eval.py \
    --model_path ./checkpoints/aomt_mixed/epoch_4 \
    --config_path $CONFIG --tokenizer $TOKENIZER \
    --benchmark alfworld --n_episodes $N \
    --output_file results/quick_eval/aomt_mixed_alfworld.json \
    2>&1 | tee results/quick_eval/aomt_mixed.log &
PID3=$!

# GPU 7: Prefix SFT Stage 2
CUDA_VISIBLE_DEVICES=7 python3 eval/task_eval.py \
    --model_path ./checkpoints/prefix_sft_s2/epoch_2 \
    --config_path $CONFIG --tokenizer $TOKENIZER \
    --benchmark alfworld --n_episodes $N \
    --output_file results/quick_eval/prefix_sft_s2_alfworld.json \
    2>&1 | tee results/quick_eval/prefix_sft_s2.log &
PID4=$!

echo "PIDs: SFT=$PID1  AOMT-Act=$PID2  AOMT-Mix=$PID3  Prefix-S2=$PID4"
echo "Waiting for all to finish..."

wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=========================================="
echo "  SUMMARY (5 ALFWorld episodes each)"
echo "=========================================="
for name in sft_standard aomt_action_only aomt_mixed prefix_sft_s2; do
    echo -n "  $name: "
    cat "results/quick_eval/${name}_alfworld.json" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'success_rate={d.get(\"success_rate\",\"?\")}')" 2>/dev/null || echo "N/A"
done
echo "Done @ $(date)"
