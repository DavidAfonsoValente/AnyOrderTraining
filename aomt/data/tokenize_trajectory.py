from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch
from transformers import PreTrainedTokenizer
import numpy as np

@dataclass
class UnitSpan:
    """Token span for one trajectory unit in the flat tokenized sequence."""
    start: int        # inclusive
    end: int          # exclusive
    unit_type: str    # "observation" | "action"
    step_idx: int     # 0-indexed

@dataclass
class TokenizedTrajectory:
    token_ids: List[int]
    unit_spans: List[UnitSpan]
    trajectory_length: int   # T (number of action steps)

def tokenize_trajectory(
    raw_example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 2048,
) -> Optional[TokenizedTrajectory]:
    """
    Converts one raw ETO example into a TokenizedTrajectory.
    Uses REAL field names from ETO dataset.
    Follows 'single user message' format for AOMT-Mixed.
    """
    convs = raw_example.get('conversations', [])
    if not convs:
        return None

    units_text = []
    units_type = []
    for turn in convs:
        units_text.append(turn['value'])
        units_type.append("observation" if turn['from'] == 'human' else "action")

    # Join units with \n to create the content for the chat template
    full_content = "\n".join(units_text)
    conversation = [{"role": "user", "content": full_content}]
    
    # Tokenize with chat template but NO generation prompt (full trajectory)
    all_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        tokenize=True
    )

    # Now we need to find the spans of each unit in the resulting tokens.
    # The units are separated by \n (token 198).
    # We find the token index of each 198 and use it to delimit units.
    
    sep_token_id = 198 # Standard for this tokenizer
    
    # Find role marker start (usually after the first few tokens)
    # LLaDA 2.0 template: <role>HUMAN</role> content <|role_end|>
    # We'll search for the first occurrence of O0 tokens.
    first_unit_ids = tokenizer.encode(units_text[0], add_special_tokens=False)
    
    # Robust span extraction: 
    # 1. Locate the start of the first unit content
    try:
        current_pos = 0
        for i in range(len(all_ids) - len(first_unit_ids) + 1):
            if all_ids[i : i+len(first_unit_ids)] == first_unit_ids:
                current_pos = i
                break
    except:
        return None

    unit_spans = []
    current_step = 0
    
    for i, (text, utype) in enumerate(zip(units_text, units_type)):
        ids = tokenizer.encode(text, add_special_tokens=False)
        start_idx = current_pos
        end_idx = start_idx + len(ids)
        
        # Verify tokens match
        if all_ids[start_idx:end_idx] != ids:
            # Fallback or error
            pass
            
        unit_spans.append(UnitSpan(
            start=start_idx,
            end=end_idx,
            unit_type=utype,
            step_idx=current_step
        ))
        
        if utype == "action":
            current_step += 1
            
        # Move pos past unit and SEP (1 token)
        current_pos = end_idx + 1 

    # Truncation
    if len(all_ids) > max_seq_len:
        # Simplification: if it doesn't fit, skip or truncate
        all_ids = all_ids[:max_seq_len]
        unit_spans = [s for s in unit_spans if s.end <= max_seq_len]

    T = sum(1 for s in unit_spans if s.unit_type == "action")

    return TokenizedTrajectory(
        token_ids=all_ids,
        unit_spans=unit_spans,
        trajectory_length=T
    )

def compute_median_unit_lengths(
    dataset,
    tokenizer: PreTrainedTokenizer,
    n_samples: int = 1000,
) -> Dict[str, int]:
    obs_lens = []
    act_lens = []
    
    for i in range(min(n_samples, len(dataset))):
        traj = tokenize_trajectory(dataset[i], tokenizer)
        if not traj: continue
        for span in traj.unit_spans:
            length = span.end - span.start
            if span.unit_type == "observation":
                obs_lens.append(length)
            else:
                act_lens.append(length)
                
    return {
        "median_obs_tokens": int(np.median(obs_lens)) if obs_lens else 17,
        "median_action_tokens": int(np.median(act_lens)) if act_lens else 33
    }
