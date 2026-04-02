import torch
import numpy as np
from typing import List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import ceil

def tokenize_history(
    history: List[str],
    tokenizer: PreTrainedTokenizer,
    sep_token: str = "\n",
) -> torch.Tensor:
    """
    Tokenizes a history list [O0, A0, O1, A1, ..., Ot] into a flat token
    sequence using the SAME format as the training dataset.
    
    CRITICAL: last unit does NOT get a trailing SEP to match tokenize_trajectory logic.
    """
    all_ids = []
    for i, text in enumerate(history):
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if i < len(history) - 1:
            sep_ids = tokenizer.encode(sep_token, add_special_tokens=False)
            all_ids.extend(sep_ids)
    
    return torch.tensor(all_ids, dtype=torch.long)

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

@torch.no_grad()
def block_diffusion_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    prompt_len: int,
    mask_token_id: int,
    diffusion_steps: int = 32,
    temperature: float = 0.0,
) -> torch.Tensor:
    """
    Confidence-based iterative unmasking (block diffusion).
    Only positions from [prompt_len:] are candidates for unmasking.
    """
    device = input_ids.device
    remaining = [i for i in range(prompt_len, input_ids.shape[0]) if input_ids[i] == mask_token_id]
    
    for step in range(diffusion_steps):
        if not remaining:
            break
            
        outputs = model(input_ids.unsqueeze(0))
        logits = outputs.logits[0]
        
        curr_mask_indices = torch.tensor(remaining, device=device)
        n_to_unmask = ceil(len(remaining) / (diffusion_steps - step))
        
        probs = torch.softmax(logits[curr_mask_indices], dim=-1)
        confidences, predictions = torch.max(probs, dim=-1)
        
        top_k_indices = torch.topk(confidences, min(n_to_unmask, len(confidences))).indices
        
        for idx in top_k_indices:
            pos = remaining[idx]
            input_ids[pos] = predictions[idx]
            
        remaining = [r for j, r in enumerate(remaining) if j not in top_k_indices.tolist()]
        
    return input_ids

def generate_next_action_mode_a(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    history: List[str],
    max_new_tokens: int = 256,
    diffusion_steps: int = 32,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """
    Mode A: Myopic next-action inference.
    """
    prompt_ids = tokenize_history(history, tokenizer).to(device)
    prompt_len = len(prompt_ids)
    
    mask_ids = torch.full((max_new_tokens,), tokenizer.mask_token_id, dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, mask_ids])
    
    denoised_ids = block_diffusion_decode(
        model, input_ids, prompt_len, tokenizer.mask_token_id, 
        diffusion_steps, temperature
    )
    
    action_tokens = denoised_ids[prompt_len:]
    return tokenizer.decode(action_tokens, skip_special_tokens=True).strip()

def generate_next_action_mode_b(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    history: List[str],
    method: str = "aomt_mixed",
    planning_horizon: int = 3,
    median_action_tokens: int = 32,
    median_obs_tokens: int = 64,
    diffusion_steps: int = 32,
    temperature: float = 0.0,
    device: str = "cuda",
) -> str:
    """
    Mode B: Planning inference. ONLY valid for aomt_mixed.
    """
    if method != "aomt_mixed":
        raise ValueError(f"Mode B is ONLY valid for aomt_mixed, got {method}")
    
    if planning_horizon < 1:
        raise ValueError(f"planning_horizon must be >= 1, got {planning_horizon}")

    prompt_ids = tokenize_history(history, tokenizer).to(device)
    prompt_len = len(prompt_ids)
    
    template_ids = [prompt_ids]
    sep_ids = torch.tensor(tokenizer.encode("\n", add_special_tokens=False), dtype=torch.long, device=device)
    mask_token_id = tokenizer.mask_token_id
    
    at_start = prompt_len
    # Need to add a SEP before the first masked action because tokenize_history doesn't add one to the last unit
    template_ids.append(sep_ids)
    # Update prompt_len to include the separator
    prompt_len += len(sep_ids)
    at_start = prompt_len
    at_end = at_start + median_action_tokens
    
    for h in range(planning_horizon):
        # Action slot
        template_ids.append(torch.full((median_action_tokens,), mask_token_id, dtype=torch.long, device=device))
        template_ids.append(sep_ids)
        # Observation slot
        template_ids.append(torch.full((median_obs_tokens,), mask_token_id, dtype=torch.long, device=device))
        if h < planning_horizon - 1:
            template_ids.append(sep_ids)
            
    input_ids = torch.cat(template_ids)
    
    denoised_ids = block_diffusion_decode(
        model, input_ids, prompt_len, mask_token_id,
        diffusion_steps, temperature
    )
    
    action_tokens = denoised_ids[at_start:at_end]
    return tokenizer.decode(action_tokens, skip_special_tokens=True).strip()
