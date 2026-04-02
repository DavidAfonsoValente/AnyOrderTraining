from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
from transformers import PreTrainedTokenizer
import numpy as np

@dataclass
class UnitSpan:
    """Token span for one trajectory unit in the flat tokenized sequence."""
    start: int        # inclusive
    end: int          # exclusive
    unit_type: str    # "observation" | "action"
    step_idx: int     # which t

@dataclass
class TokenizedTrajectory:
    token_ids: List[int]
    unit_spans: List[UnitSpan]
    trajectory_length: int   # T (number of action steps)

def tokenize_trajectory(
    raw_example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 2048,
    sep_token: str = "\n"
) -> Optional[TokenizedTrajectory]:
    """
    Converts one raw ETO dataset example into a TokenizedTrajectory.
    """
    # From Phase 1 inspection: 'conversations' contains [{'from': 'human', 'value': '...'}, ...]
    convs = raw_example.get('conversations', [])
    if not convs:
        return None

    # ETO AlfWorld format: usually O0, OK, O_goal, ...
    # Based on the inspection, human=obs, gpt=act. 
    # But often there's a system prompt at start.
    
    units_text = []
    units_type = []
    
    for turn in convs:
        units_text.append(turn['value'])
        units_type.append("observation" if turn['from'] == 'human' else "action")

    # Interleave with sep_token
    all_token_ids = []
    unit_spans = []
    
    # Track steps. Each action increments the step index for the next units.
    current_step = 0
    
    for i, (text, utype) in enumerate(zip(units_text, units_type)):
        # Tokenize this unit
        ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Add separator if not the last unit
        if i < len(units_text) - 1:
            sep_ids = tokenizer.encode(sep_token, add_special_tokens=False)
            ids = ids + sep_ids
            
        start_idx = len(all_token_ids)
        all_token_ids.extend(ids)
        end_idx = len(all_token_ids)
        
        unit_spans.append(UnitSpan(
            start=start_idx,
            end=end_idx,
            unit_type=utype,
            step_idx=current_step
        ))
        
        if utype == "action":
            current_step += 1

    # Truncation from the END
    if len(all_token_ids) > max_seq_len:
        while len(all_token_ids) > max_seq_len and len(unit_spans) > 1:
            last_span = unit_spans.pop()
            all_token_ids = all_token_ids[:last_span.start]
        
        if not unit_spans:
            return None

    # T is the number of actions
    T = sum(1 for s in unit_spans if s.unit_type == "action")

    return TokenizedTrajectory(
        token_ids=all_token_ids,
        unit_spans=unit_spans,
        trajectory_length=T
    )

def compute_median_unit_lengths(
    dataset,
    tokenizer,
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
        "median_obs_tokens": int(np.median(obs_lens)) if obs_lens else 64,
        "median_action_tokens": int(np.median(act_lens)) if act_lens else 32
    }
