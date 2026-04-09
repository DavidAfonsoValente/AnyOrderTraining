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

def _ensure_list_of_ints(token_output: Any) -> List[int]:
    """Helper to handle LLaDA tokenizer returning Encoding objects."""
    if hasattr(token_output, "ids"):
        return token_output.ids
    if isinstance(token_output, list):
        return [int(i) for i in token_output]
    return list(token_output)

def tokenize_trajectory(
    raw_example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 2048,
) -> Optional[TokenizedTrajectory]:
    """
    Converts one raw ETO example into a TokenizedTrajectory.
    """
    convs = raw_example.get('conversations', [])
    if not convs: return None

    units_text = [turn['value'] for turn in convs]
    units_type = ["observation" if turn['from'] == 'human' else "action" for turn in convs]

    full_content = "\n".join(units_text)
    conversation = [{"role": "user", "content": full_content}]
    
    # Tokenize and ensure we have a list of ints
    raw_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True)
    all_ids = _ensure_list_of_ints(raw_ids)

    # Search for the content tokens to find the start of O0
    first_unit_ids = tokenizer.encode(units_text[0], add_special_tokens=False)
    current_pos = 0
    for i in range(len(all_ids) - len(first_unit_ids) + 1):
        if all_ids[i : i+len(first_unit_ids)] == first_unit_ids:
            current_pos = i
            break

    unit_spans = []
    current_step = 0
    for text, utype in zip(units_text, units_type):
        ids = tokenizer.encode(text, add_special_tokens=False)
        start_idx = current_pos
        end_idx = start_idx + len(ids)
        unit_spans.append(UnitSpan(start_idx, end_idx, utype, current_step))
        if utype == "action": current_step += 1
        current_pos = end_idx + 1 # +1 for \n

    if len(all_ids) > max_seq_len:
        all_ids = all_ids[:max_seq_len]
        unit_spans = [s for s in unit_spans if s.end <= max_seq_len]

    return TokenizedTrajectory(all_ids, unit_spans, sum(1 for s in unit_spans if s.unit_type == "action"))

def compute_median_unit_lengths(dataset, tokenizer, n_samples=1000) -> Dict[str, int]:
    obs_lens, act_lens = [], []
    for i in range(min(n_samples, len(dataset))):
        traj = tokenize_trajectory(dataset[i], tokenizer)
        if not traj: continue
        for span in traj.unit_spans:
            length = span.end - span.start
            if span.unit_type == "observation": obs_lens.append(length)
            else: act_lens.append(length)
    return {
        "median_obs_tokens": int(np.median(obs_lens)) if obs_lens else 17,
        "median_action_tokens": int(np.median(act_lens)) if act_lens else 33
    }
