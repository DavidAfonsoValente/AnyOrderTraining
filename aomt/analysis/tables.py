import pandas as pd
import os
import json
import glob

def generate_main_results_table(results_dir, output_dir):
    """TABLE 1: main_results.csv"""
    methods = {
        "std_sft": "Standard SFT",
        "prefix_s1": "Prefix SFT (IWM)",
        "prefix_s2": "Prefix SFT Stage 2",
        "amx_p025": "AOMT-Mixed (p=0.25)"
    }
    rows = []
    for m_key, m_name in methods.items():
        res_files = glob.glob(os.path.join(results_dir, "eval", m_key, "*.json"))
        m_results = {
            "Method": m_name,
            "ALFWorld U%": 0.0,
            "SciWorld U": 0.0,
            "WebShop Rew": 0.0,
            "Grad steps/traj": "1" if "amx" in m_key else "T"
        }
        for f in res_files:
            if "rho0.0" not in f: continue
            with open(f, "r") as src:
                data = json.load(src)
                if "alfworld" in data: m_results["ALFWorld U%"] = data["alfworld"].get("success_rate", 0.0)
                if "scienceworld" in data: m_results["SciWorld U"] = data["scienceworld"].get("all_score", 0.0)
                if "webshop" in data: m_results["WebShop Rew"] = data["webshop"].get("reward", 0.0)
        rows.append(m_results)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "main_results.csv"), index=False)

def generate_robustness_table(results_dir, output_dir):
    """TABLE 4: robustness_results.csv"""
    rhos = [0.0, 0.1, 0.2, 0.3]
    methods = {
        "std_sft": "Standard SFT",
        "amx_p025": "AOMT-Mixed (p=0.25)"
    }
    rows = []
    for m_key, m_name in methods.items():
        m_row = {"Method": m_name}
        for rho in rhos:
            f_path = os.path.join(results_dir, "eval", m_key, f"results_alfworld_rho{rho}.json")
            val = 0.0
            if os.path.exists(f_path):
                with open(f_path, "r") as src:
                    data = json.load(src)
                    val = data.get("alfworld", {}).get("success_rate", 0.0)
            m_row[f"rho={rho}"] = val
        # Retention = Success(rho=0.3) / Success(rho=0.0)
        if m_row["rho=0.0"] > 0:
            m_row["Retention"] = m_row["rho=0.3"] / m_row["rho=0.0"]
        else:
            m_row["Retention"] = 0.0
        rows.append(m_row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "robustness_results.csv"), index=False)

def generate_ksweep_table(results_dir, output_dir):
    """TABLE 3: k_sweep_results.csv"""
    k_steps = [1, 2, 4, 8, 16, 32, 64]
    rows = []
    for k in k_steps:
        pattern = os.path.join(results_dir, "eval", "amx_p025", "ksweep", f"k{k}", "*.json")
        res_files = glob.glob(pattern)
        val = 0.0
        if res_files:
            with open(res_files[0], "r") as f:
                data = json.load(f)
                val = data.get("alfworld", {}).get("success_rate", 0.0)
        rows.append({"K steps": k, "ALFWorld Success%": val})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "ksweep_results.csv"), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_main_results_table(args.results_dir, args.output_dir)
    generate_ksweep_table(args.results_dir, args.output_dir)
    generate_robustness_table(args.results_dir, args.output_dir)
