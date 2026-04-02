from huggingface_hub import snapshot_download
import os

model_id = "inclusionAI/LLaDA2.0-mini"
local_dir = "aomt/weights/LLaDA2.0-mini"

print(f"Downloading {model_id} to {local_dir}...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
)
print("Download complete.")
