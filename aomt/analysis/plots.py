import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_method_comparison(results_dir, output_dir):
    """FIGURE 1: method_comparison_bar.png"""
    methods = ["Std SFT", "Prefix SFT", "AOMT-Act", "AOMT-Mix (A)", "AOMT-Mix (B)"]
    success = [0, 0, 0, 0, 0]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, success)
    plt.title("ALFWorld Unseen Success Rate")
    plt.ylabel("Success (%)")
    plt.savefig(os.path.join(output_dir, "method_comparison_bar.png"))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_method_comparison(args.results_dir, args.output_dir)
