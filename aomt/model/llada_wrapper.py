import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple

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
    
    if tokenizer.mask_token_id == tokenizer.pad_token_id:
        # Some tokenizers use same for both, but for AOMT we prefer distinct if possible.
        # But we'll follow whatever the model was trained with.
        pass

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

# Typing helper
from typing import Any
