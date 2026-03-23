#!/bin/bash
set -euo pipefail

AOMT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$AOMT_ROOT"

mkdir -p logs checkpoints weights results data/cache

PARENT_DIR="$(dirname "$AOMT_ROOT")"
export PYTHONPATH="$AOMT_ROOT/dFactory/VeOmni:$AOMT_ROOT/dFactory:$PARENT_DIR:${PYTHONPATH:-}"
export AOMT_ROOT

NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_LAUNCH_BLOCKING=0

export HF_HOME=/scratch/e1507650/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/e1507650/.cache/huggingface/hub
export HF_DATASETS_CACHE=/scratch/e1507650/.cache/huggingface/datasets
export PIP_CACHE_DIR=/scratch/e1507650/.cache/pip
export WANDB_DIR=/scratch/e1507650/.cache/wandb
export XDG_CACHE_HOME=/scratch/e1507650/.cache
export ALFWORLD_DATA=/scratch/e1507650/.cache/alfworld
export ALFWORLD_CONFIG="$AOMT_ROOT/configs/alfworld_config.yaml"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$WANDB_DIR"

echo "=== AOMT Pipeline (Direct Execution) ==="
echo "  AOMT_ROOT: $AOMT_ROOT"
echo "  GPUs: $NUM_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  HF_HOME: $HF_HOME"
echo "  Date: $(date)"
echo ""

START_STEP=0
if [[ "${1:-}" == "--from" ]]; then
    START_STEP="${2:-0}"
    echo "  Resuming from step $START_STEP"
fi

run_step() {
    local STEP_NUM="$1"
    local STEP_NAME="$2"
    if (( STEP_NUM < START_STEP )); then
        echo "[SKIP] Step $STEP_NUM: $STEP_NAME"
        return 0
    fi
    echo ""
    echo "========================================================"
    echo "[Step $STEP_NUM] $STEP_NAME"
    echo "  Started: $(date)"
    echo "========================================================"
}

# STEP 0: Model Preparation
run_step 0 "Model Preparation"
if (( 0 >= START_STEP )); then
    MODELS_DIR="$AOMT_ROOT/weights"
    ORIGINAL_MODEL_PATH="$MODELS_DIR/LLaDA2.0-mini"
    MERGED_MODEL_PATH="$MODELS_DIR/llada2-mini-merged"

    if [ -z "$(ls -A "$ORIGINAL_MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
        echo "Downloading LLaDA 2.0-mini weights from HuggingFace..."
        python "$AOMT_ROOT/dFactory/scripts/download_hf_model.py" --repo_id "inclusionAI/LLaDA2.0-mini" --local_dir "$MODELS_DIR"
    else
        echo "Model weights already present."
    fi

    if [ ! -d "$MERGED_MODEL_PATH" ] || [ -z "$(ls -A "$MERGED_MODEL_PATH"/*.safetensors 2>/dev/null)" ]; then
        echo "Converting MoE to merged-expert format..."
        rm -rf "$MERGED_MODEL_PATH" 2>/dev/null || true
        python "$AOMT_ROOT/dFactory/scripts/moe_convertor.py" \
            --input-path "$ORIGINAL_MODEL_PATH" --output-path "$MERGED_MODEL_PATH" --mode merge
        cp "$AOMT_ROOT/dFactory/models/llada2_moe/modeling_llada2_moe.py" "$MERGED_MODEL_PATH"/
        cp "$AOMT_ROOT/dFactory/models/llada2_moe/configuration_llada2_moe.py" "$MERGED_MODEL_PATH"/
        cp "$AOMT_ROOT/dFactory/models/llada2_moe/parallel_plan.py" "$MERGED_MODEL_PATH"/
        cp "$ORIGINAL_MODEL_PATH"/*.json "$MERGED_MODEL_PATH"/ 2>/dev/null || true
        cp "$ORIGINAL_MODEL_PATH"/*.model "$MERGED_MODEL_PATH"/ 2>/dev/null || true
        sed -i 's/"model_type": "llada2_moe"/"model_type": "llada2_moe_veomni"/' "$MERGED_MODEL_PATH/config.json"
    else
        echo "Merged model already exists."
    fi
fi

# STEP 1: Data Preparation
run_step 1 "Data Preparation"
if (( 1 >= START_STEP )); then
    python data/measure_lengths.py --tokenizer ./weights/LLaDA2.0-mini --gen_length 256 --max_seq_len 2048 | tee logs/length_analysis.txt
    python data/prepare_data.py --output_dir ./data/cache/ --tokenizer ./weights/LLaDA2.0-mini
    ls -lh data/cache/
fi

# STEP 2: Standard SFT
run_step 2 "Training: Standard SFT (3 epochs)"
if (( 2 >= START_STEP )); then
    mkdir -p checkpoints/sft_standard
    torchrun --standalone --nproc_per_node="$NUM_GPUS" tasks/train_standard_sft.py configs/sft_standard.yaml 2>&1 | tee logs/02_sft_standard.log
fi

# STEP 3: Prefix SFT Stage 1
run_step 3 "Training: Prefix SFT Stage 1"
if (( 3 >= START_STEP )); then
    mkdir -p checkpoints/prefix_sft_s1
    torchrun --standalone --nproc_per_node="$NUM_GPUS" tasks/train_standard_sft.py configs/prefix_sft_stage1.yaml 2>&1 | tee logs/03_prefix_s1.log
fi

# STEP 4: Prefix SFT Stage 2 (needs Stage 1 ckpt prepared as model dir)
run_step 4 "Training: Prefix SFT Stage 2"
if (( 4 >= START_STEP )); then
    STAGE1_MERGED="./weights/prefix_sft_s1_merged"
    if [ ! -d "$STAGE1_MERGED" ] || [ -z "$(ls -A "$STAGE1_MERGED"/*.safetensors 2>/dev/null)" ]; then
        echo "Preparing Stage 1 checkpoint as model dir for Stage 2..."
        BEST_S1=$(ls -td checkpoints/prefix_sft_s1/epoch_*/ 2>/dev/null | head -1)
        mkdir -p "$STAGE1_MERGED"
        for f in "$BEST_S1"/model-*.safetensors "$BEST_S1"/model.safetensors.index.json; do
            ln -sf "$(realpath "$f")" "$STAGE1_MERGED/$(basename "$f")"
        done
        cp weights/llada2-mini-merged/config.json "$STAGE1_MERGED/"
        cp weights/llada2-mini-merged/modeling_llada2_moe.py "$STAGE1_MERGED/"
        cp weights/llada2-mini-merged/configuration_llada2_moe.py "$STAGE1_MERGED/"
        cp weights/llada2-mini-merged/parallel_plan.py "$STAGE1_MERGED/"
        cp weights/llada2-mini-merged/*.json "$STAGE1_MERGED/" 2>/dev/null || true
        cp weights/llada2-mini-merged/*.model "$STAGE1_MERGED/" 2>/dev/null || true
    fi
    mkdir -p checkpoints/prefix_sft_s2
    torchrun --standalone --nproc_per_node="$NUM_GPUS" tasks/train_standard_sft.py configs/prefix_sft_stage2.yaml 2>&1 | tee logs/04_prefix_s2.log
fi

# STEP 5: AOMT Action-Only
run_step 5 "Training: AOMT Action-Only (5 epochs)"
if (( 5 >= START_STEP )); then
    mkdir -p checkpoints/aomt_action_only
    torchrun --standalone --nproc_per_node="$NUM_GPUS" tasks/train_aomt.py configs/aomt_action_only.yaml 2>&1 | tee logs/05_aomt_action.log
fi

# STEP 6: AOMT Mixed
run_step 6 "Training: AOMT Mixed (5 epochs)"
if (( 6 >= START_STEP )); then
    mkdir -p checkpoints/aomt_mixed
    torchrun --standalone --nproc_per_node="$NUM_GPUS" tasks/train_aomt.py configs/aomt_mixed.yaml 2>&1 | tee logs/06_aomt_mixed.log
fi

# STEP 7: Task Evaluation
run_step 7 "Evaluation: Task Success Rates"
if (( 7 >= START_STEP )); then
    convert_ckpt() {
        local NAME="$1" CKPT_DIR="$2"
        if [ -d "./weights/${NAME}-sep" ] && [ -n "$(ls -A "./weights/${NAME}-sep"/*.safetensors 2>/dev/null)" ]; then
            echo "Converted weights for ${NAME} already exist, skipping."
            return
        fi
        BEST=$(ls -td "$CKPT_DIR"/epoch_*/ 2>/dev/null | head -1)
        if [ -z "$BEST" ]; then echo "WARNING: No checkpoint in $CKPT_DIR"; return; fi
        echo "Converting ${NAME} from ${BEST}..."
        cp ./weights/llada2-mini-merged/config.json "$BEST/" 2>/dev/null || true
        cp ./weights/llada2-mini-merged/configuration_llada2_moe.py "$BEST/" 2>/dev/null || true
        cp ./weights/llada2-mini-merged/modeling_llada2_moe.py "$BEST/" 2>/dev/null || true
        cp ./weights/llada2-mini-merged/tokenizer*.json "$BEST/" 2>/dev/null || true
        cp ./weights/llada2-mini-merged/special_tokens_map.json "$BEST/" 2>/dev/null || true
        python dFactory/scripts/moe_convertor.py --input-path "$BEST" --output-path "./weights/${NAME}-sep" --mode split
        cp ./weights/LLaDA2.0-mini/modeling_llada2_moe.py "./weights/${NAME}-sep/" 2>/dev/null || true
    }
    convert_ckpt sft_standard checkpoints/sft_standard
    convert_ckpt prefix_sft_s2 checkpoints/prefix_sft_s2
    convert_ckpt aomt_action checkpoints/aomt_action_only
    convert_ckpt aomt_mixed checkpoints/aomt_mixed

    EVAL_TASKS=()
    for MODEL in sft_standard prefix_sft_s2 aomt_action aomt_mixed; do
        for BENCH in alfworld scienceworld; do
            EVAL_TASKS+=("${MODEL}|${BENCH}")
        done
    done

    GPU_IDS=(1 2 3)
    NUM_SLOTS=${#GPU_IDS[@]}
    PIDS=()
    SLOTS=()
    IDX=0

    launch_eval() {
        local gpu_slot=$1 task=$2
        local MODEL="${task%%|*}" BENCH="${task##*|}"
        local cuda_dev="${GPU_IDS[$gpu_slot]}"
        echo "[GPU cuda:$cuda_dev] Launching: $MODEL / $BENCH"
        CUDA_VISIBLE_DEVICES="$cuda_dev" python eval/task_eval.py \
            --model_path "./weights/${MODEL}-sep" --tokenizer ./weights/LLaDA2.0-mini \
            --benchmark "$BENCH" --split test --n_episodes 50 \
            --gen_length 128 --block_length 32 --steps 16 \
            --device "cuda:0" \
            --output_file "results/${MODEL}_${BENCH}.json" \
            > "logs/eval_${MODEL}_${BENCH}.log" 2>&1 &
    }

    for (( i=0; i<NUM_SLOTS && IDX<${#EVAL_TASKS[@]}; i++ )); do
        launch_eval "$i" "${EVAL_TASKS[$IDX]}"
        PIDS+=("$!")
        SLOTS+=("$i")
        IDX=$((IDX+1))
    done

    while [ ${#PIDS[@]} -gt 0 ]; do
        for j in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$j]}" 2>/dev/null; then
                wait "${PIDS[$j]}" || echo "WARNING: eval task on slot ${SLOTS[$j]} exited with error (check logs/)"
                FREE_SLOT="${SLOTS[$j]}"
                unset 'PIDS[j]'
                unset 'SLOTS[j]'
                PIDS=("${PIDS[@]}")
                SLOTS=("${SLOTS[@]}")
                if (( IDX < ${#EVAL_TASKS[@]} )); then
                    launch_eval "$FREE_SLOT" "${EVAL_TASKS[$IDX]}"
                    PIDS+=("$!")
                    SLOTS+=("$FREE_SLOT")
                    IDX=$((IDX+1))
                fi
                break
            fi
        done
        sleep 5
    done
    echo "All evaluations complete."
fi

# STEP 8: NLL-obs
run_step 8 "Evaluation: Observation-Masked NLL"
if (( 8 >= START_STEP )); then
    for MODEL in aomt_action aomt_mixed; do
        python eval/nll_obs.py --model_path "./weights/${MODEL}-sep" --tokenizer ./weights/LLaDA2.0-mini \
            --data_path ./data/cache/aomt_test.jsonl --output_file "results/${MODEL}_nll_obs.json"
    done
fi

# STEP 9: Noise Robustness
run_step 9 "Evaluation: Noise Robustness (ALFWorld)"
if (( 9 >= START_STEP )); then
    for RHO in 0.1 0.2 0.3; do
        for MODEL in sft_standard aomt_action aomt_mixed; do
            python eval/noise_robustness.py --model_path "./weights/${MODEL}-sep" --tokenizer ./weights/LLaDA2.0-mini \
                --benchmark alfworld --split seen --rho "$RHO" --output_file "results/${MODEL}_robustness_rho${RHO}.json"
        done
    done
fi

# STEP 10: Collate Results
run_step 10 "Collating Final Results"
if (( 10 >= START_STEP )); then
    python eval/collate_results.py --results_dir results/ | tee results/summary_table.txt
fi

echo ""
echo "========================================================"
echo "  PIPELINE COMPLETE — $(date)"
echo "  Results: results/summary_table.txt"
echo "========================================================"
