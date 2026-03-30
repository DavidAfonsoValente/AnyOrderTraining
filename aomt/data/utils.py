# aomt/data/utils.py
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
import os

def load_robust_dataset(dataset_name="agent-eto/eto-sft-trajectory"):
    """
    Robustly loads the agent-eto/eto-sft-trajectory dataset, 
    handling heterogeneous schema issues (inconsistent columns and types).
    Returns (combined_split_ds, benchmark_test_splits)
    """
    if dataset_name != "agent-eto/eto-sft-trajectory":
        return load_dataset(dataset_name), {}

    print(f"Applying robust loading for {dataset_name}...")
    base_url = "https://huggingface.co/datasets/agent-eto/eto-sft-trajectory/resolve/main/data"
    
    # Load individual files
    alfworld_ds = load_dataset("json", data_files=f"{base_url}/alfworld_sft.json", split="train")
    sciworld_ds = load_dataset("json", data_files=f"{base_url}/sciworld_sft.json", split="train")
    webshop_ds = load_dataset("json", data_files=f"{base_url}/webshop_sft.json", split="train")

    # Standardize schemas
    def standardize(ds):
        # Remove inconsistent columns
        cols = [c for c in ['reward', 'source', 'variation'] if c in ds.column_names]
        if cols: ds = ds.remove_columns(cols)
        # Cast ID to string
        if 'id' in ds.features: ds = ds.cast_column('id', Value(dtype='string'))
        return ds

    alfworld_ds = standardize(alfworld_ds)
    sciworld_ds = standardize(sciworld_ds)
    webshop_ds = standardize(webshop_ds)

    # Individual splits for benchmark-specific NLL_obs
    b_splits = {
        "alfworld":     alfworld_ds.train_test_split(test_size=0.1, seed=42)["test"],
        "scienceworld": sciworld_ds.train_test_split(test_size=0.1, seed=42)["test"],
        "webshop":      webshop_ds.train_test_split(test_size=0.1, seed=42)["test"]
    }

    # Merge
    combined = concatenate_datasets([alfworld_ds, sciworld_ds, webshop_ds])
    split_ds = combined.train_test_split(test_size=0.1, seed=42)
    
    return split_ds, b_splits
