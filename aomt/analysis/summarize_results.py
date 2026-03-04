# analysis/summarize_results.py
import json
import glob
import os
import pandas as pd

def find_and_summarize_results(checkpoints_dir: str):
    """
    Finds all 'results.json' files within a directory, aggregates them,
    and prints a summary table.
    """
    search_path = os.path.join(checkpoints_dir, "**", "results.json")
    results_files = glob.glob(search_path, recursive=True)

    if not results_files:
        print(f"No 'results.json' files found in '{checkpoints_dir}'.")
        print("Please run the evaluation script first.")
        return

    all_data = []
    for file in results_files:
        with open(file, 'r') as f:
            data = json.load(f)
            
            # Extract the experiment name from the path
            # e.g., ./checkpoints/aomt_mixed/results.json -> aomt_mixed
            experiment_name = os.path.basename(os.path.dirname(file))
            data['experiment'] = experiment_name
            all_data.append(data)

    # Create a pandas DataFrame for easy manipulation and display
    df = pd.DataFrame(all_data)
    
    # Define the columns we care about for the summary table
    # This can be adjusted to include more metrics
    summary_columns = [
        'experiment',
        'avg_success_rate',
        'mean_nll_obs',
        'mean_nll_act',
        'success_rate_rho0', # No noise
        'success_rate_rho1', # 10% noise
        'success_rate_rho2', # 20% noise
        'success_rate_rho3', # 30% noise
    ]
    
    # Filter for columns that actually exist in the dataframe
    existing_columns = [col for col in summary_columns if col in df.columns]
    
    summary_df = df[existing_columns].set_index('experiment')
    
    # Format the output for better readability
    summary_df = summary_df.round(4)
    
    print("========================================")
    print("        Experiment Results Summary")
    print("========================================")
    
    # pd.set_option('display.max_rows', 500)
    pd.set_option('display.width', 120)

    print(summary_df)
    
    print("
========================================")

def main():
    checkpoints_dir = "checkpoints"
    # In the project structure, checkpoints are in the parent dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_path = os.path.join(project_root, checkpoints_dir)
    
    find_and_summarize_results(checkpoints_path)

if __name__ == "__main__":
    main()
