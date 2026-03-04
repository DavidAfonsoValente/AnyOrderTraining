# aomt/data/download.py
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
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
            print("Applying special handling for 'agent-eto/eto-sft-trajectory' dataset due to heterogeneous data files.")
            
            # Load each data file as a separate dataset
            print("Loading individual data files using direct HTTPS URLs...")
            base_url = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/data"
            alfworld_ds = load_dataset("json", data_files=f"{base_url}/alfworld_sft.json", split="train", download_mode="force_redownload")
            sciworld_ds = load_dataset("json", data_files=f"{base_url}/sciworld_sft.json", split="train", download_mode="force_redownload")
            webshop_ds = load_dataset("json", data_files=f"{base_url}/webshop_sft.json", split="train", download_mode="force_redownload")

            # The 'webshop' file contains extra columns ('reward', 'source') that are
            # not present in the other files, causing schema mismatches.
            # These columns are not used by the parsing scripts, so we remove them.
            columns_to_remove = [c for c in ['reward', 'source'] if c in webshop_ds.column_names]
            if columns_to_remove:
                print(f"Removing inconsistent columns from 'webshop' data: {columns_to_remove}")
                webshop_ds = webshop_ds.remove_columns(columns_to_remove)

            # Fix for 'id' column type mismatch before concatenation
            all_datasets = [alfworld_ds, sciworld_ds, webshop_ds]
            for i, ds in enumerate(all_datasets):
                if 'id' in ds.features and not isinstance(ds.features['id'], Value):
                    print(f"Casting 'id' column of dataset {i} to string type.")
                    all_datasets[i] = ds.cast_column('id', Value(dtype='string'))

            # Concatenate all parts into a single 'train' split
            print("Concatenating data files into a single 'train' split...")
            train_dataset = concatenate_datasets(all_datasets)
            
            # The processing script expects a 'train' split. It can handle a missing 'test' split.
            dataset = DatasetDict({ "train": train_dataset })

        else:
            # For any other dataset, use the default loading behavior
            dataset = load_dataset(DATASET_NAME)
        
        print(f"Saving dataset to '{save_path}'...")
        dataset.save_to_disk(save_path)
        print(f"Dataset successfully downloaded and saved to '{save_path}'.")
    except Exception as e:
        print(f"An error occurred while processing the dataset: {e}")
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
