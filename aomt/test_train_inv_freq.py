import sys, torch, os, yaml
sys.path.insert(0, 'dFactory/VeOmni')
sys.path.insert(0, 'dFactory')
from veomni.models import build_foundation_model
from veomni.distributed.parallel_state import init_parallel_state, get_parallel_state
import torch.distributed as dist

os.environ["VEOMNI_USE_LIGER_KERNEL"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

dist.init_process_group(backend="nccl")
init_parallel_state(dp_size=1, dp_mode="fsdp2")

with open("configs/sft_standard.yaml", "r") as f:
    config = yaml.safe_load(f)

model = build_foundation_model(
    weights_path=config["model"]["model_path"],
    config_path=config["model"]["model_path"],
    torch_dtype="bfloat16",
    attn_implementation="sdpa", 
    init_device="meta",
    moe_implementation="fused"
)

from veomni.distributed.torch_parallelize import build_parallelize_model
model = build_parallelize_model(
    model, 
    weights_path=config["model"]["model_path"],
    enable_gradient_checkpointing=False,
    enable_mixed_precision=False,
    basic_modules=["LLaDA2MoeDecoderLayer"],
    init_device="meta"
)

inv_freq = model.model.rotary_emb.inv_freq
print(f"Training script inv_freq: min={inv_freq.min().item():.6f}, max={inv_freq.max().item():.6f}, all_zeros={bool((inv_freq==0).all())}")
