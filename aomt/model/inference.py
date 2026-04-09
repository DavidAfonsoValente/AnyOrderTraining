import torch
import numpy as np
from typing import List, Optional, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import ceil

def _ensure_list_of_ints(token_output: Any) -> List[int]:
    """Universal flattener for LLaDA Tokenizer outputs."""
    if hasattr(token_output, "get") and "input_ids" in token_output:
        return _ensure_list_of_ints(token_output["input_ids"])
    if hasattr(token_output, "data") and isinstance(token_output.data, dict):
        if "input_ids" in token_output.data:
            return _ensure_list_of_ints(token_output.data["input_ids"])
    if hasattr(token_output, "ids"):
        return [int(i) for i in token_output.ids]
    if torch.is_tensor(token_output):
        return token_output.flatten().tolist()
    if isinstance(token_output, list):
        if len(token_output) > 0:
            if isinstance(token_output[0], (list, dict)) or hasattr(token_output[0], "ids"):
                return _ensure_list_of_ints(token_output[0])
        return [int(i) for i in token_output]
    return [int(token_output)]

def generate_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    history_parts: List[str],
    gen_length: int = 256,
    steps: int = 32,
    temperature: float = 0.0,
) -> str:
    """
    Identical for ALL methods.
    """
    prompt = "\n".join(history_parts) + "\n"
    conversation = [{"role": "user", "content": prompt}]
    
    # 1. Get prompt tokens
    raw_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True)
    input_ids = _ensure_list_of_ints(raw_ids)
    
    role_end_id = int(input_ids[-1])
    prompt_ids = input_ids[:-1]
    prompt_len = len(prompt_ids)
    
    # 2. Append MASK block and trailing role end
    full_ids = prompt_ids + [int(tokenizer.mask_token_id)] * gen_length + [role_end_id]
    input_tensors = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    
    # 3. Native generate()
    with torch.no_grad():
        output_ids = model.generate(
            input_tensors,
            gen_length=gen_length,
            steps=steps,
            temperature=int(temperature),
            mask_id=int(tokenizer.mask_token_id),
            eos_id=int(tokenizer.eos_token_id)
        )
    
    # 4. Extract generation slot
    generated = output_ids[0, prompt_len:prompt_len + gen_length]
    
    # Clean up generation
    eos_pos = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0: generated = generated[:eos_pos[0]]
    
    # Robust newline stopping: only stop if newline is NOT the very first token
    raw_nl = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = int(_ensure_list_of_ints(raw_nl)[0])
    
    nl_pos = (generated == newline_id).nonzero(as_tuple=True)[0]
    if len(nl_pos) > 0:
        # If the model started with a newline, we skip it and look for the next one
        if nl_pos[0] == 0 and len(nl_pos) > 1:
            generated = generated[1:nl_pos[1]]
        elif nl_pos[0] > 0:
            generated = generated[:nl_pos[0]]
        
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def corrupt_observation(
    observation: str,
    tokenizer: PreTrainedTokenizer,
    rho: float,
    rng: np.random.Generator,
) -> str:
    if rho <= 0: return observation
    token_ids = _ensure_list_of_ints(tokenizer.encode(observation, add_special_tokens=False))
    n_tokens = len(token_ids)
    n_to_corrupt = int(round(rho * n_tokens))
    if n_to_corrupt > 0:
        indices = rng.choice(n_tokens, size=n_to_corrupt, replace=False)
        for idx in indices:
            token_ids[idx] = int(rng.integers(0, tokenizer.vocab_size))
    return tokenizer.decode(token_ids)
