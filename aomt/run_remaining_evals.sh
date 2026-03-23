#!/bin/bash
set -e
cd /scratch/e1507650/AnyOrderTraining/aomt

export HF_HOME=/scratch/e1507650/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/e1507650/.cache/huggingface
export ALFWORLD_DATA=/scratch/e1507650/.cache/alfworld
export ALFWORLD_CONFIG="$PWD/configs/alfworld_config.yaml"

mkdir -p logs results

REMAINING_TASKS=(
    "aomt_action|alfworld"
    "aomt_mixed|alfworld"
    "sft_standard|scienceworld"
    "prefix_sft_s2|scienceworld"
    "aomt_action|scienceworld"
    "aomt_mixed|scienceworld"
)

GPU_IDS=(1 2 3)

launch_eval() {
    local gpu_id=$1 task=$2
    local MODEL="${task%%|*}" BENCH="${task##*|}"
    echo "[GPU cuda:$gpu_id] Launching: $MODEL / $BENCH"
    CUDA_VISIBLE_DEVICES="$gpu_id" python eval/task_eval.py \
        --model_path "./weights/${MODEL}-sep" --tokenizer ./weights/LLaDA2.0-mini \
        --benchmark "$BENCH" --split test --n_episodes 50 \
        --gen_length 128 --block_length 32 --steps 16 \
        --device "cuda:0" \
        --output_file "results/${MODEL}_${BENCH}.json" \
        > "logs/eval_${MODEL}_${BENCH}.log" 2>&1 &
}

wait_for_free_gpu() {
    while true; do
        for gpu_id in "${GPU_IDS[@]}"; do
            local used
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
            if [ "$used" -lt 1000 ]; then
                echo "$gpu_id"
                return
            fi
        done
        sleep 10
    done
}

IDX=0
PIDS=()
TASK_NAMES=()
GPU_USED=()

# First pass: launch one per free GPU
for gpu_id in "${GPU_IDS[@]}"; do
    if [ $IDX -ge ${#REMAINING_TASKS[@]} ]; then break; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
    if [ "$used" -lt 1000 ]; then
        launch_eval "$gpu_id" "${REMAINING_TASKS[$IDX]}"
        PIDS+=($!)
        TASK_NAMES+=("${REMAINING_TASKS[$IDX]}")
        GPU_USED+=("$gpu_id")
        IDX=$((IDX + 1))
    fi
done

echo "Launched ${#PIDS[@]} initial tasks. ${#REMAINING_TASKS[@]} total, $IDX dispatched so far."

# Wait loop: when a task finishes, launch next on its GPU
while [ ${#PIDS[@]} -gt 0 ]; do
    for j in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$j]}" 2>/dev/null; then
            wait "${PIDS[$j]}" 2>/dev/null
            EC=$?
            if [ $EC -ne 0 ]; then
                echo "WARNING: ${TASK_NAMES[$j]} exited with code $EC (check logs/)"
            else
                echo "DONE: ${TASK_NAMES[$j]}"
            fi
            FREE_GPU="${GPU_USED[$j]}"
            unset 'PIDS[j]' 'TASK_NAMES[j]' 'GPU_USED[j]'
            PIDS=("${PIDS[@]}")
            TASK_NAMES=("${TASK_NAMES[@]}")
            GPU_USED=("${GPU_USED[@]}")

            if [ $IDX -lt ${#REMAINING_TASKS[@]} ]; then
                sleep 5
                launch_eval "$FREE_GPU" "${REMAINING_TASKS[$IDX]}"
                PIDS+=($!)
                TASK_NAMES+=("${REMAINING_TASKS[$IDX]}")
                GPU_USED+=("$FREE_GPU")
                IDX=$((IDX + 1))
            fi
            break
        fi
    done
    sleep 5
done

echo ""
echo "=== All remaining evaluations complete ==="
echo "Results:"
ls -la results/*.json 2>/dev/null || echo "(no results yet)"
