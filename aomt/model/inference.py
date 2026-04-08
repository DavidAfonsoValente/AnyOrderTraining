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
    Identical for ALL methods.
    history_parts: list of obs/act strings ending with latest observation.
    Joined as single user message — matches Standard SFT and AOMT training format.
    
    To resolve the training distribution mismatch, we format the prompt exactly
    as the model saw it during training (a single USER message), removing the
    trailing <|role_end|> token, appending the generation slot (MASKs), and
    then appending <|role_end|>. The model denoises inside the USER role.
    """
    prompt = "\n".join(history_parts) + "\n"
    conversation = [{"role": "user", "content": prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True,
    )
    
    # Remove trailing <|role_end|> to generate inside the user message
    # We find the role_end token ID manually or assume it's the last token.
    # For LLaDA 2.0, the chat template ends with <|role_end|>
    role_end_id = input_ids[-1]
    prompt_ids = input_ids[:-1]
    
    prompt_len = len(prompt_ids)
    
    mask_ids = [tokenizer.mask_token_id] * gen_length
    
    # Re-append role_end_id after masks
    full_ids = prompt_ids + mask_ids + [role_end_id]
    
    input_tensors = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_tensors,
            gen_length=gen_length,
            block_length=32,
            steps=steps,
            temperature=temperature,
            cfg_scale=0.0,
            remasking="low_confidence",
        )
    
    generated = output_ids[0, prompt_len:prompt_len + gen_length]
    
    # Handle EOS
    eos_pos = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        generated = generated[:eos_pos[0]]
    
    # Stop generation at the first newline (since actions are single lines in ETO format)
    # The tokenizer encoding for newline is typically 198
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    nl_pos = (generated == newline_id).nonzero(as_tuple=True)[0]
    if len(nl_pos) > 0:
        generated = generated[:nl_pos[0]]
        
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
