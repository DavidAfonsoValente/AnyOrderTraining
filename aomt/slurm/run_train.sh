#!/bin/bash
# aomt/slurm/run_train.sh
# Launch ALL training jobs for the AOMT paper.
# Usage: bash aomt/slurm/run_train.sh
set -euo pipefail

echo "════════════════════════════════════════"
echo "AOMT Training Pipeline — Submitting Jobs"
echo "════════════════════════════════════════"

# ── Independent training jobs ──────────────────────────────────────────────
J_SFT=$(sbatch --parsable aomt/slurm/train_standard_sft.sh)
J_IWM=$(sbatch --parsable aomt/slurm/train_prefix_sft_stage1.sh)

# AOMT-Mixed p sweep
J_AMX_P015=$(sbatch --parsable aomt/slurm/train_aomt_mixed_p015.sh)
J_AMX_P025=$(sbatch --parsable aomt/slurm/train_aomt_mixed_p025.sh)
J_AMX_P040=$(sbatch --parsable aomt/slurm/train_aomt_mixed_p040.sh)
J_AMX_P050=$(sbatch --parsable aomt/slurm/train_aomt_mixed_p050.sh)

# Ablation A2: token-level masking training run
J_AMX_TOKEN=$(sbatch --parsable aomt/slurm/train_aomt_mixed_token_level.sh)

# ── Stage 2 depends on Stage 1 ─────────────────────────────────────────────
J_SFT2=$(sbatch --parsable --dependency=afterok:$J_IWM \
         aomt/slurm/train_prefix_sft_stage2.sh)

# ── Collect all training job IDs ───────────────────────────────────────────
ALL_TRAIN="${J_SFT}:${J_IWM}:${J_SFT2}"
ALL_TRAIN="${ALL_TRAIN}:${J_AMX_P015}:${J_AMX_P025}:${J_AMX_P040}:${J_AMX_P050}"
ALL_TRAIN="${ALL_TRAIN}:${J_AMX_TOKEN}"

# Write all job IDs for use by run_eval_and_analysis.sh
mkdir -p aomt/slurm
echo "$ALL_TRAIN" > aomt/slurm/.train_ids

# ── Print summary ──────────────────────────────────────────────────────────
echo ""
echo "Training jobs submitted:"
printf "  %-40s Job %s\n" "standard_sft:"             "$J_SFT"
printf "  %-40s Job %s\n" "prefix_sft_stage1:"         "$J_IWM"
printf "  %-40s Job %s\n" "prefix_sft_stage2:"         "$J_SFT2 (after $J_IWM)"
printf "  %-40s Job %s\n" "aomt_mixed_p sweep:"        "$J_AMX_P015 $J_AMX_P025 $J_AMX_P040 $J_AMX_P050"
printf "  %-40s Job %s\n" "aomt_mixed_token_level:"    "$J_AMX_TOKEN"
echo ""
echo "All train IDs written to: aomt/slurm/.train_ids"
echo "When training completes, run: bash aomt/slurm/run_eval_and_analysis.sh"
echo "Monitor: squeue -u \$USER"
