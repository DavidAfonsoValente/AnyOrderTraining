# aomt/data/utils.py
from datasets import load_dataset, DatasetDict, concatenate_datasets, Value
import os

def load_robust_dataset(dataset_name="agent-eto/eto-sft-trajectory"):
    """
    Robustly loads the agent-eto/eto-sft-trajectory dataset, 
    handling heterogeneous schema issues (inconsistent columns and types).
    """
    if dataset_name != "agent-eto/eto-sft-trajectory":
        return load_dataset(dataset_name)

    print(f"Applying robust loading for {dataset_name}...")
    base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main/data"
    
    # Load individual files
    alfworld_ds = load_dataset("json", data_files=f"{base_url}/alfworld_sft.json", split="train")
    sciworld_ds = load_dataset("json", data_files=f"{base_url}/sciworld_sft.json", split="train")
    webshop_ds = load_dataset("json", data_files=f"{base_url}/webshop_sft.json", split="train")

    # Remove inconsistent columns from webshop
    cols_to_remove = [c for c in ['reward', 'source'] if c in webshop_ds.column_names]
    if cols_to_remove:
        webshop_ds = webshop_ds.remove_columns(cols_to_remove)

    # Standardize 'id' column to string
    all_ds = [alfworld_ds, sciworld_ds, webshop_ds]
    for i, ds in enumerate(all_ds):
        if 'id' in ds.features:
            all_ds[i] = ds.cast_column('id', Value(dtype='string'))

    # Merge
    combined = concatenate_datasets(all_ds)
    
    # Split into train/test (90/10) as the original dataset doesn't have a clean test split on HF
    split_ds = combined.train_test_split(test_size=0.1, seed=42)
    
    return split_ds
