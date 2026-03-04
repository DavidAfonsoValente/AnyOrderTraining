# aomt/data/download.py
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import argparse

# The dataset name as specified in the engineering specs
DATASET_NAME = "agent-eto/eto-sft-trajectory"
DEFAULT_SAVE_PATH = "./dataset_cache"

def download_dataset(save_path: str):
    """
    Downloads the agent-eto/eto-sft-trajectory dataset from Hugging Face and
    saves it to a local directory.

    Args:
        save_path (str): The directory to save the dataset to.
    """
    if os.path.exists(save_path):
        print(f"Dataset directory '{save_path}' already exists. Skipping download.")
        print("If you want to re-download, please remove the existing directory.")
        return

    print(f"Downloading dataset '{DATASET_NAME}'...")
    try:
        if DATASET_NAME == "agent-eto/eto-sft-trajectory":
            print("Applying special handling for 'agent-eto/eto-sft-trajectory' dataset...")
            
            # We use the 'json' loader and pass the repo name to 'data_files' 
            # as a dictionary or use the 'path' parameter correctly.
            
            print("Loading gworld_sft.json...")
            gworld_ds = load_dataset("json", data_files=f"hf://datasets/{DATASET_NAME}/data/gworld_sft.json", split="train")
            
            print("Loading mind2web_sft.json...")
            mind2web_ds = load_dataset("json", data_files=f"hf://datasets/{DATASET_NAME}/data/mind2web_sft.json", split="train")
            
            print("Loading webshop_sft.json...")
            webshop_ds = load_dataset("json", data_files=f"hf://datasets/{DATASET_NAME}/data/webshop_sft.json", split="train")

            # Handle the column mismatch
            columns_to_remove = [c for c in ['reward', 'source'] if c in webshop_ds.column_names]
            if columns_to_remove:
                print(f"Removing inconsistent columns from 'webshop': {columns_to_remove}")
                webshop_ds = webshop_ds.remove_columns(columns_to_remove)

            print("Concatenating into single 'train' split...")
            train_dataset = concatenate_datasets([gworld_ds, mind2web_ds, webshop_ds])
            dataset = DatasetDict({"train": train_dataset})

        else:
            dataset = load_dataset(DATASET_NAME)

def main():
    """Main function to run the download script."""
    parser = argparse.ArgumentParser(
        description=f"Download the {DATASET_NAME} dataset from Hugging Face."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=DEFAULT_SAVE_PATH,
        help=f"The local directory to save the dataset. Defaults to '{DEFAULT_SAVE_PATH}'.",
    )
    args = parser.parse_args()
    
    # Ensure the path is relative to the script's directory for consistency
    script_dir = os.path.dirname(__file__)
    save_path = os.path.join(script_dir, args.save_path)

    download_dataset(save_path)

if __name__ == "__main__":
    main()
