import argparse
import os
from .analysis.tables import generate_main_results_table
from .analysis.plots import plot_method_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--format", type=str, default="both", choices=["latex", "csv", "both"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating tables...")
    generate_main_results_table(args.results_dir, args.output_dir)
    
    print("Generating plots...")
    plot_method_comparison(args.results_dir, args.output_dir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
