import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.modeling_rope_utils as mru
from typing import Tuple, Any

# Monkey-patch nn.Module to ensure set_submodule is available for bitsandbytes
if not hasattr(nn.Module, "set_submodule"):
    def set_submodule(self, target: str, module: nn.Module) -> None:
        if not target:
            raise ValueError("Target path cannot be empty.")
        parts = target.split(".")
        target_mod = self.get_submodule(".".join(parts[:-1]))
        setattr(target_mod, parts[-1], module)
    nn.Module.set_submodule = set_submodule

# 1. ROPE Patch: LLaDA 2.0-mini expects 'default' in ROPE_INIT_FUNCTIONS
# which was removed in newer transformers versions.
def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    base = getattr(config, "rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return inv_freq, 1.0

if "default" not in mru.ROPE_INIT_FUNCTIONS:
    mru.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

LLADA_MODEL_ID = "aomt/weights/LLaDA2.0-mini"

def load_model_and_tokenizer(
    model_id: str = LLADA_MODEL_ID,
    precision: str = "bf16",
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
) -> Tuple[torch.nn.Module, Any]:
    """
    Loads LLaDA 2.0-mini model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer MUST have a mask_token_id for AOMT.")
    
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    return model, tokenizer
