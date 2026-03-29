#!/bin/bash
#SBATCH --job-name=aomt_maskprob_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_maskprob_sweep_%j.out
#SBATCH --error=logs/eval_maskprob_sweep_%j.err

set -euo pipefail
export PYTHONPATH=.

RESULTS_DIR="eval/results/maskprob_sweep"
mkdir -p "${RESULTS_DIR}" logs

echo "[$(date)] Evaluating mask_prob sweep on ALFWorld (seen) using 1x A100-80GB"

for MASK_PROB in 0.15 0.25 0.40 0.50; do
    CKPT="outputs/aomt_mixed_p${MASK_PROB}/epoch_2" 
    if [ ! -d "${CKPT}" ]; then
        echo "WARNING: checkpoint not found: ${CKPT} — skipping"
        continue
    fi
    echo "[$(date)] Evaluating p=${MASK_PROB} from ${CKPT}"
    python3 eval/task_eval.py \
        --method          aomt_mixed \
        --checkpoint_dir  "${CKPT}" \
        --benchmark       alfworld \
        --split           seen \
        --output_json     "${RESULTS_DIR}/aomt_mixed_p${MASK_PROB}_alfworld_seen.json" \
        2>&1 | tee "${RESULTS_DIR}/aomt_mixed_p${MASK_PROB}_alfworld_seen.log"
done

# Summary table
python3 - <<'EOF'
import json, glob, os
results_dir = "eval/results/maskprob_sweep"
print(f"\n{'mask_prob':>12} | {'ALFWorld Seen':>14}")
print("-" * 32)
for p in [0.15, 0.25, 0.40, 0.50]:
    f = f"{results_dir}/aomt_mixed_p{p}_alfworld_seen.json"
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"{p:>12.2f} | {d.get('success_rate', 0.0):>13.1f}%")
    else:
        print(f"{p:>12.2f} | {'MISSING':>14}")
EOF

echo "[$(date)] mask_prob sweep evaluation complete"
