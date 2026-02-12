# Running on SoC Slurm Cluster - Complete Guide

## Initial Setup (One Time Only)

### 1. Login to Cluster
```bash
ssh dvalente@xlogin0.comp.nus.edu.sg
# or xlogin1.comp.nus.edu.sg or xlogin2.comp.nus.edu.sg
```

### 2. Upload and Extract Project
```bash
# Upload the tar.gz file (from your local machine)
scp any-order-training.tar.gz dvalente@xlogin0.comp.nus.edu.sg:~

# On the cluster
cd ~
tar -xzf any-order-training.tar.gz
cd any-order-training
```

### 3. Run Setup Script
```bash
# Make scripts executable
chmod +x cluster_setup.sh *.sh

# Run setup (creates venv and installs dependencies)
bash cluster_setup.sh
```

This will take 5-10 minutes and create a virtual environment with all dependencies.

## Testing Phase

### Step 1: Test Data Generation (5-10 min)
```bash
sbatch test_datagen.sh
```

Monitor:
```bash
squeue -u dvalente
tail -f logs/datagen_test_*.out
```

Verify:
```bash
ls data/test/BabyAI-GoToRedBall-v0/train/*.json | wc -l
# Should show ~70 files
```

### Step 2: Test Pipeline (10-15 min)
```bash
sbatch test_pipeline.sh
```

Monitor:
```bash
tail -f logs/test_pipeline_*.out
```

Verify:
```bash
ls outputs/test_pipeline_*/checkpoints/
# Should see checkpoint files
```

## Full Experiment Run

### Step 1: Generate Full Dataset (1-2 hours)
```bash
sbatch generate_full_data.sh
```

Monitor:
```bash
squeue -j <jobid>  # Get jobid from sbatch output
tail -f logs/datagen_full_*.out
```

Verify completion:
```bash
ls data/raw/BabyAI-GoToRedBall-v0/train/*.json | wc -l
# Should show ~960 files
ls data/raw/BabyAI-GoToRedBall-v0/val/*.json | wc -l
# Should show ~120 files
ls data/raw/BabyAI-GoToRedBall-v0/test/*.json | wc -l
# Should show ~120 files
```

### Step 2: Submit All Experiments (12+ hours per experiment)
```bash
# Wait for data generation to complete, then:
bash submit_all_experiments.sh
```

This submits 6 experiments:
- Exp 1a, 1b, 1c (masking prob 0.15, 0.30, 0.50)
- Exp 2a, 2b (cell vs attribute)
- Exp 3 (scheduled masking)

Monitor all jobs:
```bash
# Check queue
squeue -u dvalente

# Watch continuously (updates every 30 seconds)
watch -n 30 'squeue -u dvalente'

# Check specific job
squeue -j <jobid>

# View output (live)
tail -f logs/exp1_p015_*.out
tail -f logs/exp2_cell_*.out
```

### Step 3: Wait for Completion
Experiments will run 12+ hours each. You'll receive email notifications when they complete.

Check completion:
```bash
# List all experiment outputs
ls -lh outputs/ablations/

# Check which have finished (have checkpoint files)
for dir in outputs/ablations/exp*/; do
    echo "$dir: $(ls $dir/checkpoints/*.pt 2>/dev/null | wc -l) checkpoints"
done
```

### Step 4: Evaluate Results
Once experiments complete:
```bash
sbatch evaluate_all.sh
```

This evaluates all experiments and generates a summary.

View results:
```bash
# After evaluation completes
tail -100 logs/eval_all_*.out

# View individual results
cat outputs/ablations/exp1_cell_p015/results.json
cat outputs/ablations/exp2_cell/results.json
```

## Monitoring Commands

### Check Job Status
```bash
# Your jobs
squeue -u dvalente

# All jobs
squeue

# Specific job details
scontrol show job <jobid>

# Job priority
sprio -j <jobid>
```

### Check Cluster Status
```bash
# Node status
sinfo

# Detailed node info
sinfo -Nel
```

### View Logs
```bash
# List all logs
ls -lth logs/

# View specific log
tail -f logs/exp1_p015_*.out

# View last 50 lines
tail -50 logs/exp2_cell_*.out

# Search for errors
grep -i error logs/*.out
```

### Cancel Jobs
```bash
# Cancel specific job
scancel <jobid>

# Cancel all your jobs
scancel -u dvalente

# Cancel specific experiment
scancel -n exp1_p015
```

## File Management

### Check Disk Usage
```bash
# Your home directory usage
du -sh ~

# Project size
du -sh ~/any-order-training

# Largest directories
du -h ~/any-order-training | sort -rh | head -20
```

### Cleanup Old Files
```bash
# Remove test outputs
rm -rf outputs/test_*

# Remove old logs
find logs -name "*.out" -mtime +7 -delete
find logs -name "*.err" -mtime +7 -delete

# Archive old experiments
tar -czf old_experiments_$(date +%Y%m%d).tar.gz outputs/ablations/
```

## Troubleshooting

### Job Pending Too Long
```bash
# Check why
squeue -j <jobid> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Check your priority
sprio -u dvalente

# Solution: Request fewer resources or different partition
```

### Out of Memory Error
```bash
# In logs: "Out of memory" or "OOM"
# Solution: Increase memory in script
#SBATCH --mem=64G  # instead of 32G

# Or reduce batch size
--data.batch_size 4  # instead of 8
```

### GPU Not Available
```bash
# Check GPU availability
sinfo -o "%20N %10c %10m %25f %10G"

# Request specific GPU type
#SBATCH --gpus=a100:1
```

### Virtual Environment Issues
```bash
# Recreate venv
rm -rf venv
bash cluster_setup.sh
```

## Quick Reference

### File Locations
```
~/any-order-training/
├── data/
│   ├── test/          # Test data (~100 trajectories)
│   └── raw/           # Full data (~1200 trajectories)
├── outputs/
│   ├── test_pipeline_*/   # Test runs
│   └── ablations/         # Experiment results
├── logs/              # Slurm output logs
└── venv/              # Python virtual environment
```

### Important Scripts
```bash
cluster_setup.sh           # Initial setup
test_datagen.sh           # Test data generation
test_pipeline.sh          # Pipeline test
generate_full_data.sh     # Full data generation
submit_all_experiments.sh # Submit all experiments
evaluate_all.sh           # Evaluate results
exp1a_p015.sh            # Individual experiments
exp1b_p030.sh
exp1c_p050.sh
exp2a_cell.sh
exp2b_attribute.sh
exp3_scheduled.sh
```

### Typical Workflow
```bash
# 1. Setup (once)
bash cluster_setup.sh

# 2. Test (20 min)
sbatch test_datagen.sh && sleep 60 && sbatch test_pipeline.sh

# 3. Generate data (2 hours)
sbatch generate_full_data.sh

# 4. Run experiments (12+ hours)
bash submit_all_experiments.sh

# 5. Evaluate (2 hours)
sbatch evaluate_all.sh

# 6. Download results
scp -r dvalente@xlogin0.comp.nus.edu.sg:~/any-order-training/outputs/ablations .
```

## Getting Results

### View Summary
```bash
# After evaluate_all.sh completes
tail -100 logs/eval_all_*.out
```

### Download Results to Local Machine
```bash
# From your local machine
scp -r dvalente@xlogin0.comp.nus.edu.sg:~/any-order-training/outputs/ablations .
scp -r dvalente@xlogin0.comp.nus.edu.sg:~/any-order-training/logs .
```

### Generate Report
```bash
# On cluster
cat > generate_report.py << 'EOF'
import json
import os

print("# Ablation Study Results\n")
print("| Experiment | NLL | Obs Acc | Act Acc |")
print("|------------|-----|---------|---------|")

for exp in sorted(os.listdir('outputs/ablations/')):
    results_file = f'outputs/ablations/{exp}/results.json'
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        nll = data.get('world_model_nll', 'N/A')
        obs = data.get('observation_accuracy', 'N/A')
        act = data.get('action_accuracy', 'N/A')
        
        print(f"| {exp:30s} | {nll:5.3f} | {obs:7.3f} | {act:7.3f} |")
EOF

python generate_report.py > RESULTS.md
cat RESULTS.md
```

## Contact & Support

If you encounter issues:
1. Check logs: `tail -100 logs/<jobname>_*.out`
2. Check the cluster status: `sinfo`
3. Email CF support or consult the SoC compute cluster documentation

## Summary

**Total Time**: ~2-3 days
- Setup: 10 min
- Testing: 30 min
- Data generation: 2 hours
- Experiments: 12+ hours (running in parallel)
- Evaluation: 2 hours

**Expected Results**:
- 6 trained models
- Comprehensive evaluation metrics
- Publishable ablation study results
