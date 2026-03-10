#!/usr/bin/env python3
import json
import os
import glob

def select_winner():
    results_files = glob.glob("results/ablation_p*_alfworld.json")
    if not results_files:
        print("0.25") # Fallback default
        return

    best_rate = -1.0
    best_prob = "0.25"

    for fpath in results_files:
        try:
            # Extract prob from filename: ablation_p0_15_alfworld.json -> 0.15
            prob_str = os.path.basename(fpath).split('_')[1][1:].replace('_', '.')
            
            with open(fpath, 'r') as f:
                data = json.load(f)
                rate = data.get("success_rate", 0.0)
                
            if rate > best_rate:
                best_rate = rate
                best_prob = prob_str
        except Exception:
            continue

    print(best_prob)

if __name__ == "__main__":
    select_winner()
