import torch
import numpy as np
from typing import List, Optional, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import ceil

def _ensure_list_of_ints(token_output: Any) -> List[int]:
    """Helper to handle LLaDA tokenizer returning Encoding or BatchEncoding objects."""
    if isinstance(token_output, dict) or hasattr(token_output, "data"):
        if "input_ids" in token_output:
            return _ensure_list_of_ints(token_output["input_ids"])
    if hasattr(token_output, "ids"):
        return [int(i) for i in token_output.ids]
    if isinstance(token_output, list):
        if len(token_output) > 0:
            if isinstance(token_output[0], list) or hasattr(token_output[0], "ids") or isinstance(token_output[0], dict):
                return _ensure_list_of_ints(token_output[0])
        return [int(i) for i in token_output]
    if torch.is_tensor(token_output):
        return token_output.flatten().tolist()
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
    
    # 3. Create 4D attention mask [B, 1, L, L]
    seq_len = len(full_ids)
    attention_mask = torch.ones((1, 1, seq_len, seq_len), device=model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensors,
            attention_mask=attention_mask,
            gen_length=gen_length,
            block_length=32,
            steps=steps,
            temperature=temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    
    # 4. Extract and clean generation
    generated = output_ids[0, prompt_len:prompt_len + gen_length]
    
    eos_pos = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0: generated = generated[:eos_pos[0]]
    
    raw_nl = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = int(_ensure_list_of_ints(raw_nl)[0])
    
    nl_pos = (generated == newline_id).nonzero(as_tuple=True)[0]
    if len(nl_pos) > 0: generated = generated[:nl_pos[0]]
        
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def corrupt_observation(
    observation: str,
    tokenizer: PreTrainedTokenizer,
    rho: float,
    rng: np.random.Generator,
) -> str:
    if rho <= 0: return observation
    raw_ids = tokenizer.encode(observation, add_special_tokens=False)
    token_ids = _ensure_list_of_ints(raw_ids)
    n_tokens = len(token_ids)
    n_to_corrupt = int(round(rho * n_tokens))
    if n_to_corrupt > 0:
        indices = rng.choice(n_tokens, size=n_to_corrupt, replace=False)
        for idx in indices:
            token_ids[idx] = int(rng.integers(0, tokenizer.vocab_size))
    return tokenizer.decode(token_ids)
