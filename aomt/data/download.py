# aomt/data/download.py
import os
from datasets import load_dataset
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
        # Use the save_to_disk method to store the dataset locally
        dataset = load_dataset(DATASET_NAME)
        dataset.save_to_disk(save_path)
        print(f"Dataset successfully downloaded and saved to '{save_path}'.")
    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        print("Please check your internet connection and Hugging Face Hub credentials.")

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
