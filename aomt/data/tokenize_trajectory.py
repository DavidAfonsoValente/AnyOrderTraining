from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Union
import torch
from transformers import PreTrainedTokenizer
import numpy as np

@dataclass
class UnitSpan:
    start: int
    end: int
    unit_type: str
    step_idx: int

@dataclass
class TokenizedTrajectory:
    token_ids: List[int]
    unit_spans: List[UnitSpan]
    trajectory_length: int

def _ensure_list_of_ints(token_output: Any) -> List[int]:
    """Absolute flattener. Returns only List[int]."""
    # 1. Handle Dict/BatchEncoding
    if isinstance(token_output, dict) or hasattr(token_output, "get"):
        if "input_ids" in token_output:
            return _ensure_list_of_ints(token_output["input_ids"])
    
    # 2. Handle metadata objects
    if hasattr(token_output, "ids"):
        token_output = token_output.ids
        
    # 3. Handle Tensors
    if torch.is_tensor(token_output):
        return token_output.flatten().tolist()
    
    # 4. Handle Lists/Iterables
    if isinstance(token_output, (list, tuple)):
        out = []
        for item in token_output:
            if isinstance(item, (list, dict)) or hasattr(item, "ids"):
                out.extend(_ensure_list_of_ints(item))
            else:
                try:
                    out.append(int(item))
                except:
                    pass # Skip non-integers
        return out
    
    return [int(token_output)]

def tokenize_trajectory(
    raw_example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 2048,
) -> Optional[TokenizedTrajectory]:
    convs = raw_example.get('conversations', [])
    if not convs: return None

    units_text = [turn['value'] for turn in convs]
    units_type = ["observation" if turn['from'] == 'human' else "action" for turn in convs]

    full_content = "\n".join(units_text)
    conversation = [{"role": "user", "content": full_content}]
    
    # Get base sequence
    raw_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=True)
    all_ids = _ensure_list_of_ints(raw_ids)

    # Find start of O0
    first_u_raw = tokenizer.encode(units_text[0], add_special_tokens=False)
    first_u_ids = _ensure_list_of_ints(first_u_raw)
    
    current_pos = 0
    # Search for the first unit ids in the full sequence
    found_first = False
    for i in range(len(all_ids) - len(first_u_ids) + 1):
        if all_ids[i : i+len(first_u_ids)] == first_u_ids:
            current_pos = i
            found_first = True
            break
    
    # Fallback: if strict match fails, the template might have added prefix tokens
    # Just start after the first few tokens (usually BOS + template header)
    if not found_first:
        current_pos = max(0, len(all_ids) - sum(len(_ensure_list_of_ints(tokenizer.encode(t, add_special_tokens=False))) for t in units_text) - (len(units_text) - 1))

    unit_spans = []
    current_step = 0
    for text, utype in zip(units_text, units_type):
        u_ids = _ensure_list_of_ints(tokenizer.encode(text, add_special_tokens=False))
        start_idx = current_pos
        end_idx = start_idx + len(u_ids)
        unit_spans.append(UnitSpan(start_idx, end_idx, utype, current_step))
        if utype == "action": current_step += 1
        current_pos = end_idx + 1

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
            l = span.end - span.start
            if span.unit_type == "observation": obs_lens.append(l)
            else: act_lens.append(l)
    return {"median_obs_tokens": int(np.median(obs_lens or [17])), "median_action_tokens": int(np.median(act_lens or [33]))}
