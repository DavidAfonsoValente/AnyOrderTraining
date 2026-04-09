import torch
import numpy as np
from typing import List, Optional, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import ceil

def _ensure_list_of_ints(token_output: Any) -> List[int]:
    """Absolute flattener. Returns only List[int]."""
    if isinstance(token_output, dict) or hasattr(token_output, "get"):
        if "input_ids" in token_output: return _ensure_list_of_ints(token_output["input_ids"])
    if hasattr(token_output, "ids"): token_output = token_output.ids
    if torch.is_tensor(token_output): return token_output.flatten().tolist()
    if isinstance(token_output, (list, tuple)):
        out = []
        for item in token_output:
            if isinstance(item, (list, dict)) or hasattr(item, "ids"): out.extend(_ensure_list_of_ints(item))
            else:
                try: out.append(int(item))
                except: pass
        return out
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
    Unified inference logic.
    """
    prompt = "\n".join(history_parts) + "\n"
    conversation = [{"role": "user", "content": prompt}]
    
    raw_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True)
    input_ids = _ensure_list_of_ints(raw_ids)
    
    role_end_id = int(input_ids[-1])
    prompt_ids = input_ids[:-1]
    prompt_len = len(prompt_ids)
    
    # Construction
    full_ids = prompt_ids + [int(tokenizer.mask_token_id)] * gen_length + [role_end_id]
    input_tensors = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensors,
            gen_length=gen_length,
            steps=steps,
            temperature=int(temperature),
            mask_id=int(tokenizer.mask_token_id),
            eos_id=int(tokenizer.eos_token_id)
        )
    
    generated_ids = output_ids[0, prompt_len : prompt_len + gen_length]
    
    # 1. Strip EOS
    eos_pos = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0: generated_ids = generated_ids[:eos_pos[0]]
    
    # 2. Decode and cleanup
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 3. If model started with newline, ensure we don't return empty
    # Action should be the first non-empty line
    for line in text.split("\n"):
        if line.strip(): return line.strip()
        
    return text.strip()

def corrupt_observation(
    observation: str,
    tokenizer: PreTrainedTokenizer,
    rho: float,
    rng: np.random.Generator,
) -> str:
    if rho <= 0: return observation
    ids = _ensure_list_of_ints(tokenizer.encode(observation, add_special_tokens=False))
    if not ids: return observation
    n_to_corrupt = int(round(rho * len(ids)))
    if n_to_corrupt > 0:
        indices = rng.choice(len(ids), size=n_to_corrupt, replace=False)
        for idx in indices: ids[idx] = int(rng.integers(0, tokenizer.vocab_size))
    return tokenizer.decode(ids)
