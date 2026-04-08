import torch
import numpy as np
from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import ceil

def generate_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    history_parts: List[str],
    gen_length: int = 256,
    steps: int = 32,
    temperature: float = 0.0,
) -> str:
    """
    Identical for ALL five methods.
    history_parts: list of obs/act strings ending with latest observation.
    Joined as single user message — matches Standard SFT training format.
    """
    prompt = "\n".join(history_parts)
    conversation = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        # Using the custom generate method implemented in modeling_llada2_moe.py
        output_ids = model.generate(
            input_ids,
            gen_length=gen_length,
            block_length=32,
            steps=steps,
            temperature=temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    
    generated = output_ids[0, input_ids.shape[1]:]
    
    # Handle EOS
    eos_pos = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        generated = generated[:eos_pos[0]]
    
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def corrupt_observation(
    observation: str,
    tokenizer: PreTrainedTokenizer,
    rho: float,
    rng: np.random.Generator,
) -> str:
    """
    Corrupts fraction rho of observation TOKENS with random vocab tokens.
    """
    if rho <= 0:
        return observation
        
    token_ids = tokenizer.encode(observation, add_special_tokens=False)
    n_tokens = len(token_ids)
    n_to_corrupt = int(round(rho * n_tokens))
    
    if n_to_corrupt > 0:
        indices = rng.choice(n_tokens, size=n_to_corrupt, replace=False)
        vocab_size = tokenizer.vocab_size
        for idx in indices:
            token_ids[idx] = int(rng.integers(0, vocab_size))
            
    return tokenizer.decode(token_ids)

# generate_next_action_mode_a and generate_next_action_mode_b removed per user hint.
# All methods now use generate_action.
