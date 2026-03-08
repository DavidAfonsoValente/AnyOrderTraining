#!/bin/bash
# =============================================================================
# 07_run_eval.sh — Full evaluation: task success + NLL-obs + robustness
# Runs all five models sequentially (different inference checkpoints).
# Single GPU is sufficient for eval.
# =============================================================================
#SBATCH --job-name=aomt_eval
#SBATCH --output=logs/07_eval_%j.log
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=nv:1
#SBATCH -C titanrtx    # Titan RTX (24GB) is perfect for inference
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal

set -euo pipefail
mkdir -p logs results

REPO_ROOT="$(pwd)"
PARENT_DIR="$(dirname "$REPO_ROOT")"
export PYTHONPATH="$REPO_ROOT/dFactory/VeOmni:$PARENT_DIR:${PYTHONPATH:-}"

echo "[$(date)] Starting evaluation | Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"

# --- Convert all checkpoints to sep format for inference ---------------------
convert_ckpt() {
    local NAME="$1"
    local CKPT_DIR="$2"

    if [ -d "./models/${NAME}-sep" ]; then
        echo "  Checkpoint ./models/${NAME}-sep already exists, skipping conversion."
        return
    fi

    BEST=$(ls -td "$CKPT_DIR"/*/ | head -1)
    echo "  Converting $BEST → ./models/${NAME}-sep"
    python dFactory/scripts/moe_convertor.py \
        --input-path  "${BEST}/hf_ckpt" \
        --output-path "./models/${NAME}-sep" \
        --mode split
    cp ./models/LLaDA2.0-mini/modeling_llada2_moe.py "./models/${NAME}-sep/"
}

echo "[$(date)] Converting checkpoints..."
convert_ckpt sft_standard   checkpoints/sft_standard
convert_ckpt prefix_sft_s2  checkpoints/prefix_sft_s2
convert_ckpt aomt_action    checkpoints/aomt_action_only
convert_ckpt aomt_mixed     checkpoints/aomt_mixed

# --- Task evaluation ---------------------------------------------------------
BENCHMARKS="alfworld scienceworld webshop"
MODELS="sft_standard prefix_sft_s2 aomt_action aomt_mixed"

for MODEL in $MODELS; do
    echo ""
    echo "[$(date)] Evaluating: $MODEL"
    for BENCH in $BENCHMARKS; do
        echo "  Benchmark: $BENCH"
        python eval/task_eval.py \
            --model_path "./models/${MODEL}-sep" \
            --tokenizer  "./models/LLaDA2.0-mini" \
            --benchmark  "$BENCH" \
            --gen_length 256 \
            --block_length 32 \
            --steps 32 \
            --output_file "results/${MODEL}_${BENCH}.json" \
            2>&1 | tee -a "logs/eval_${MODEL}_${BENCH}.log"
    done
done

# --- Steps=1 consistency check (AOMT-Mixed on ALFWorld) ----------------------
echo ""
echo "[$(date)] Steps=1 consistency check: AOMT-Mixed on ALFWorld..."
python eval/task_eval.py \
    --model_path ./models/aomt_mixed-sep \
    --tokenizer  ./models/LLaDA2.0-mini \
    --benchmark  alfworld \
    --gen_length 256 \
    --block_length 256 \
    --steps 1 \
    --output_file results/aomt_mixed_alfworld_steps1.json \
    2>&1 | tee -a logs/eval_aomt_mixed_alfworld_steps1.log

# --- Observation-masked NLL (AOMT models only) --------------------------------
echo ""
echo "[$(date)] Computing NLL-obs..."
for MODEL in aomt_action aomt_mixed; do
    python eval/nll_obs.py \
        --model_path "./models/${MODEL}-sep" \
        --tokenizer  ./models/LLaDA2.0-mini \
        --data_path  ./data/cache/aomt_test.jsonl \
        --output_file "results/${MODEL}_nll_obs.json"
done

# --- Robustness evaluation (ALFWorld seen, AOMT-Mixed + SFT) -----------------
echo ""
echo "[$(date)] Running robustness evaluation..."
for RHO in 0.1 0.2 0.3; do
    for MODEL in sft_standard aomt_action aomt_mixed; do
        python eval/noise_robustness.py \
            --model_path "./models/${MODEL}-sep" \
            --tokenizer  ./models/LLaDA2.0-mini \
            --benchmark  alfworld \
            --split      seen \
            --rho        "$RHO" \
            --output_file "results/${MODEL}_robustness_rho${RHO}.json"
    done
done

# --- Collate results into a summary table ------------------------------------
echo ""
echo "[$(date)] Collating results..."
python eval/collate_results.py --results_dir results/ \
    | tee results/summary_table.txt

echo ""
echo "[$(date)] Evaluation complete. Results in results/"
ls -la results/
