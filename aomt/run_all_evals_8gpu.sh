#!/bin/bash
set -e
cd /scratch/e1507650/AnyOrderTraining/aomt

export HF_HOME=/scratch/e1507650/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/e1507650/.cache/huggingface
export ALFWORLD_DATA=/scratch/e1507650/.cache/alfworld
export ALFWORLD_CONFIG="$PWD/configs/alfworld_config.yaml"
export PYTHONPATH="/scratch/e1507650/AnyOrderTraining/aomt/dFactory/VeOmni:/scratch/e1507650/AnyOrderTraining/aomt/dFactory:/scratch/e1507650/AnyOrderTraining:${PYTHONPATH:-}"

PYTHON=/scratch/e1507650/conda/envs/aomt/bin/python
export PYTHONUNBUFFERED=1

mkdir -p logs results

MODELS=(sft_standard prefix_sft_s2 aomt_action aomt_mixed)
BENCHMARKS=(alfworld scienceworld)

# GPU assignment: ALFWorld on 0-3, ScienceWorld on 4-7
# ALFWorld is ~12h, ScienceWorld is ~1-3h
TASKS=(
    "0|sft_standard|alfworld"
    "1|prefix_sft_s2|alfworld"
    "2|aomt_action|alfworld"
    "3|aomt_mixed|alfworld"
    "4|sft_standard|scienceworld"
    "5|prefix_sft_s2|scienceworld"
    "6|aomt_action|scienceworld"
    "7|aomt_mixed|scienceworld"
)

GEN_LENGTH=128
BLOCK_LENGTH=32
STEPS=16
N_EPISODES=50
SPLIT=test

echo "============================================="
echo " AOMT Full Evaluation - 8 GPU Parallel Run"
echo "============================================="
echo "Models:     ${MODELS[*]}"
echo "Benchmarks: ${BENCHMARKS[*]}"
echo "Episodes:   $N_EPISODES per eval"
echo "GPUs:       0-7 (8 total)"
echo "Start time: $(date)"
echo "============================================="

# Clear old results
rm -f results/sft_standard_alfworld.json results/prefix_sft_s2_alfworld.json
rm -f results/aomt_action_alfworld.json results/aomt_mixed_alfworld.json
rm -f results/sft_standard_scienceworld.json results/prefix_sft_s2_scienceworld.json
rm -f results/aomt_action_scienceworld.json results/aomt_mixed_scienceworld.json

PIDS=()
TASK_LABELS=()

for entry in "${TASKS[@]}"; do
    IFS='|' read -r GPU_ID MODEL BENCH <<< "$entry"
    LOGFILE="logs/eval_${MODEL}_${BENCH}.log"
    OUTFILE="results/${MODEL}_${BENCH}.json"

    echo "[GPU $GPU_ID] Launching: $MODEL / $BENCH -> $LOGFILE"

    CUDA_VISIBLE_DEVICES="$GPU_ID" $PYTHON -u eval/task_eval.py \
        --model_path "./weights/${MODEL}-sep" \
        --tokenizer ./weights/LLaDA2.0-mini \
        --benchmark "$BENCH" \
        --split "$SPLIT" \
        --n_episodes "$N_EPISODES" \
        --gen_length "$GEN_LENGTH" \
        --block_length "$BLOCK_LENGTH" \
        --steps "$STEPS" \
        --device "cuda:0" \
        --output_file "$OUTFILE" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)
    TASK_LABELS+=("GPU${GPU_ID}:${MODEL}/${BENCH}")
done

echo ""
echo "All 8 evals launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${TASK_LABELS[$i]}"
    if wait "$pid"; then
        echo "[$(date +%H:%M:%S)] DONE: $label"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $label (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================="
echo " Evaluation Complete - $(date)"
echo "============================================="
echo ""

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED eval(s) failed. Check logs/ for details."
    echo ""
fi

echo "Results:"
echo "--------"
for f in results/*_alfworld.json results/*_scienceworld.json; do
    if [ -f "$f" ]; then
        printf "  %-45s %s\n" "$(basename "$f")" "$(cat "$f")"
    fi
done

echo ""
echo "Logs:"
for f in logs/eval_*.log; do
    if [ -f "$f" ]; then
        printf "  %-45s %s\n" "$(basename "$f")" "$(wc -l < "$f") lines"
    fi
done
