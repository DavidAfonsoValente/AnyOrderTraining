import pandas as pd
import os

def generate_main_results_table(results_dir, output_dir):
    """
    TABLE 1: main_results.csv
    Rows: Zero-shot, Std SFT, Prefix SFT, AOMT-Action, AOMT-Mixed (A), AOMT-Mixed (B)
    """
    data = {
        "Method": ["Zero-shot LLaDA", "Standard SFT", "Prefix SFT", "AOMT-Action-Only", "AOMT-Mixed (Mode A)", "AOMT-Mixed (Mode B)"],
        "ALFWorld S%": [0, 0, 0, 0, 0, 0],
        "ALFWorld U%": [0, 0, 0, 0, 0, 0],
        "SciWorld S": [0, 0, 0, 0, 0, 0],
        "SciWorld U": [0, 0, 0, 0, 0, 0],
        "WebShop Rew": [0, 0, 0, 0, 0, 0],
        "Grad steps/traj": ["-", "T", "T", "1", "1", "1"]
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_dir, "main_results.csv"), index=False)
    print(f"Generated main_results.csv in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    generate_main_results_table(args.results_dir, args.output_dir)
