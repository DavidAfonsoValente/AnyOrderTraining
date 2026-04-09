#!/bin/bash
# slurm/run_part1_baselines_and_sweeps.sh
# Runs cluster GPU tests, then submits all baseline and sweep training jobs.
# Usage: bash slurm/run_part1_baselines_and_sweeps.sh
# Output: slurm/.part1_job_ids
set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
# Aligned with cluster_info.md: Use project dir or a high-quota mount
OUTPUT_ROOT=$PROJECT_DIR/results

echo "════════════════════════════════════════════════════"
echo "AOMT Part 1: Baselines + Sweeps"
echo "Project: $PROJECT_DIR"
echo "Output:  $OUTPUT_ROOT"
echo "════════════════════════════════════════════════════"

# ── Step 0: Create directories ─────────────────────────────────────────────
echo "Creating result and log directories..."
mkdir -p "$OUTPUT_ROOT/logs/"{tests,std_sft,prefix_s1,prefix_s2,amx_p015,amx_p025,amx_p040,amx_p050,amx_token,eval,ksweep,nll,ablate,analysis}
mkdir -p "$OUTPUT_ROOT/checkpoints/"{std_sft,prefix_s1,prefix_s2,amx_p015,amx_p025,amx_p040,amx_p050,amx_token}
mkdir -p "$OUTPUT_ROOT/eval"
mkdir -p "$OUTPUT_ROOT/analysis"

# ── Step 1: Submit GPU test job ────────────────────────────────────────────
echo "[1/2] Submitting GPU test pre-flight..."
J_TESTS=$(sbatch --parsable slurm/run_tests.sh)
echo "      GPU tests job: $J_TESTS"

# ── Step 2: Submit training jobs (all depend on tests passing) ─────────────
echo "[2/2] Submitting training jobs (depend on tests: $J_TESTS)..."

DEP=""

# Baselines
J_SFT=$(sbatch   --parsable $DEP slurm/train_standard_sft.sh)
J_IWM=$(sbatch   --parsable $DEP slurm/train_prefix_sft_stage1.sh)
J_SFT2=$(sbatch  --parsable --dependency=afterok:${J_IWM} \
                 slurm/train_prefix_sft_stage2.sh)

# AOMT-Mixed p_mask sweep
J_P015=$(sbatch  --parsable $DEP slurm/train_aomt_mixed_p015.sh)
J_P025=$(sbatch  --parsable $DEP slurm/train_aomt_mixed_p025.sh)
J_P040=$(sbatch  --parsable $DEP slurm/train_aomt_mixed_p040.sh)
J_P050=$(sbatch  --parsable $DEP slurm/train_aomt_mixed_p050.sh)

# Ablation A2: token-level masking
J_TOKEN=$(sbatch --parsable $DEP slurm/train_aomt_mixed_token_level.sh)

# ── Write job IDs for Part 2 ───────────────────────────────────────────────
ALL_PART1="${J_SFT}:${J_IWM}:${J_SFT2}:${J_P015}:${J_P025}:${J_P040}:${J_P050}:${J_TOKEN}"
mkdir -p "$PROJECT_DIR/slurm"
echo "$ALL_PART1" > "$PROJECT_DIR/slurm/.part1_job_ids"
echo "$J_TESTS"  >> "$PROJECT_DIR/slurm/.part1_job_ids"

echo ""
echo "Part 1 jobs submitted:"
printf "  %-35s Job %s\n" "GPU tests (pre-flight):"  "$J_TESTS"
printf "  %-35s Job %s\n" "standard_sft:"            "$J_SFT"
printf "  %-35s Job %s\n" "prefix_sft_stage1:"       "$J_IWM"
printf "  %-35s Job %s\n" "prefix_sft_stage2:"       "$J_SFT2  (after $J_IWM)"
printf "  %-35s Job %s\n" "aomt_mixed p=0.15:"       "$J_P015"
printf "  %-35s Job %s\n" "aomt_mixed p=0.25:"       "$J_P025"
printf "  %-35s Job %s\n" "aomt_mixed p=0.40:"       "$J_P040"
printf "  %-35s Job %s\n" "aomt_mixed p=0.50:"       "$J_P050"
printf "  %-35s Job %s\n" "aomt_mixed token-level:"  "$J_TOKEN"
echo ""
echo "Job IDs written to: slurm/.part1_job_ids"
echo "When Part 1 completes, run: bash slurm/run_part2_aomt_eval_and_analysis.sh"
echo "Monitor: squeue -u \$USER"
