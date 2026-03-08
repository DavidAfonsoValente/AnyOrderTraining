"""
eval/collate_results.py
Collates results from individual evaluation JSON files into a summary table.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Collate evaluation results")
    parser.add_argument("--results_dir", type=str, default="results/", help="Directory containing JSON results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Directory {results_dir} does not exist.")
        return

    data = []
    
    # Task evaluation results
    for json_file in results_dir.glob("*.json"):
        if "_nll_obs" in json_file.name or "_robustness_" in json_file.name:
            continue
            
        try:
            with open(json_file, "r") as f:
                res = json.load(f)
            
            # Expected name format: {MODEL}_{BENCHMARK}.json
            parts = json_file.stem.split("_")
            if len(parts) >= 2:
                model = "_".join(parts[:-1])
                bench = parts[-1]
                
                success_rate = res.get("success_rate", res.get("score", 0))
                data.append({
                    "Model": model,
                    "Benchmark": bench,
                    "Metric": success_rate
                })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index="Model", columns="Benchmark", values="Metric")
    
    print("\n=== Task Success / Score Summary ===")
    print(pivot_df.to_string())

    # NLL-obs results
    nll_data = []
    for json_file in results_dir.glob("*_nll_obs.json"):
        try:
            with open(json_file, "r") as f:
                res = json.load(f)
            model = json_file.stem.replace("_nll_obs", "")
            nll_data.append({"Model": model, "NLL-obs": res.get("nll_obs", 0)})
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            
    if nll_data:
        print("\n=== Observation-Masked NLL ===")
        print(pd.DataFrame(nll_data).to_string(index=False))

    # Robustness results
    rob_data = []
    for json_file in results_dir.glob("*_robustness_rho*.json"):
        try:
            with open(json_file, "r") as f:
                res = json.load(f)
            # {MODEL}_robustness_rho{RHO}.json
            model = json_file.stem.split("_robustness_")[0]
            rho = json_file.stem.split("_rho")[1]
            rob_data.append({
                "Model": model,
                "Rho": rho,
                "Success Rate": res.get("success_rate", 0)
            })
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if rob_data:
        print("\n=== Robustness Evaluation (ALFWorld) ===")
        rob_df = pd.DataFrame(rob_data)
        pivot_rob = rob_df.pivot(index="Model", columns="Rho", values="Success Rate")
        print(pivot_rob.to_string())

if __name__ == "__main__":
    main()
