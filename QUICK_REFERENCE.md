# Quick Reference Card - SoC Cluster

## Setup (One Time)
```bash
ssh dvalente@xlogin0.comp.nus.edu.sg
cd ~
tar -xzf any-order-training-cluster-ready.tar.gz
cd any-order-training
bash cluster_setup.sh  # Takes 5-10 min
```

## Test Pipeline (30 min total)
```bash
sbatch test_datagen.sh       # ~10 min
squeue -u dvalente           # Check status
sbatch test_pipeline.sh      # ~15 min (after datagen done)
```

## Run Full Experiments
```bash
# 1. Generate data (1-2 hours)
sbatch generate_full_data.sh

# 2. Wait for completion, then submit all experiments
bash submit_all_experiments.sh

# 3. Monitor (experiments run 12+ hours each)
watch -n 30 'squeue -u dvalente'

# 4. Evaluate when done
sbatch evaluate_all.sh
```

## Essential Commands

### Monitor Jobs
```bash
squeue -u dvalente              # Your jobs
squeue -j <jobid>               # Specific job
tail -f logs/exp1_p015_*.out    # Live log view
sprio -u dvalente               # Job priority
```

### Cancel Jobs
```bash
scancel <jobid>                 # Cancel one
scancel -u dvalente             # Cancel all yours
```

### Check Results
```bash
ls outputs/ablations/           # List experiments
cat outputs/ablations/exp1_cell_p015/results.json
tail -100 logs/eval_all_*.out   # Summary after evaluation
```

### Troubleshooting
```bash
# Job stuck? Check why:
squeue -j <jobid> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Out of memory? Edit script and increase:
#SBATCH --mem=64G

# GPU not available? Check:
sinfo -o "%20N %10c %10m %25f %10G"
```

## File Locations
```
data/test/          - Test data (~100 trajectories)
data/raw/           - Full data (~1200 trajectories)
outputs/ablations/  - Experiment results
logs/               - Job output logs
```

## All Scripts
```
cluster_setup.sh           - Setup environment
test_datagen.sh           - Test data generation
test_pipeline.sh          - Test training
generate_full_data.sh     - Full data generation
exp1a_p015.sh            - Experiment 1a (p=0.15)
exp1b_p030.sh            - Experiment 1b (p=0.30)
exp1c_p050.sh            - Experiment 1c (p=0.50)
exp2a_cell.sh            - Experiment 2a (cell)
exp2b_attribute.sh       - Experiment 2b (attribute)
exp3_scheduled.sh        - Experiment 3 (scheduled)
submit_all_experiments.sh - Submit all at once
evaluate_all.sh          - Evaluate all experiments
```

## Full Documentation
See `CLUSTER_WORKFLOW.md` for complete guide

## Email Updates
Change email in all .sh scripts:
```bash
sed -i 's/dvalente@comp.nus.edu.sg/YOUR_EMAIL@comp.nus.edu.sg/g' *.sh
```

## Expected Timeline
- Setup: 10 min
- Testing: 30 min  
- Data gen: 2 hours
- Training: 12-24 hours (parallel)
- Evaluation: 2 hours
- **Total**: 1-2 days
