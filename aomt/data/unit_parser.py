# aomt/data/unit_parser.py
from dataclasses import dataclass
from typing import List, Literal, Optional
import torch
from transformers import PreTrainedTokenizer

# Section 3.3: Internal Trajectory Format
@dataclass
class TrajectoryUnit:
    """A single unit in a trajectory, either an observation or an action."""
    unit_type: Literal["obs", "act"]
    text: str
    turn_index: int

@dataclass
class Trajectory:
    """Represents a complete, parsed trajectory as a sequence of units."""
    id: str
    env: str  # "alfworld", "sciworld", "webshop"
    units: List[TrajectoryUnit]

def parse_conversation_to_trajectory(example: dict) -> Trajectory:
    """
    Parses a raw conversation (from HF) or a processed AOMT example (from JSONL)
    into a structured Trajectory object.
    """
    units = []
    
    # Case 1: Processed AOMT format (unit_texts/unit_types)
    if "unit_texts" in example and "unit_types" in example:
        for i, (text, utype) in enumerate(zip(example["unit_texts"], example["unit_types"])):
            units.append(TrajectoryUnit(unit_type=utype, text=text, turn_index=i))
    
    # Case 2: Raw HF format (conversations)
    elif "conversations" in example:
        conversation_list = example["conversations"] if isinstance(example["conversations"], list) else [example["conversations"]]
        # ... (rest of the logic handles lists of lists if needed)
        if len(conversation_list) > 0 and isinstance(conversation_list[0], list):
             conversation_list = conversation_list[0]

        for i, turn in enumerate(conversation_list):
            utype = "obs" if turn["from"] == "human" else "act"
            units.append(TrajectoryUnit(unit_type=utype, text=turn["value"], turn_index=i))
    
    # Case 3: SFT format (messages)
    elif "messages" in example:
        for i, msg in enumerate(example["messages"]):
            utype = "obs" if msg["role"] == "user" else "act"
            units.append(TrajectoryUnit(unit_type=utype, text=msg["content"], turn_index=i))
    
    else:
        raise KeyError(f"Example format not recognized. Keys: {example.keys()}")

    return Trajectory(
        id=example.get("id", "unknown"), 
        env=example.get("env", example.get("task", "unknown")), 
        units=units
    )


# Section 4: Trajectory Unit Parser
@dataclass
class TokenizedUnit:
    """Records the token span for a single unit within a tokenized trajectory."""
    unit_type: Literal["obs", "act"]
    token_start: int  # inclusive index
    token_end: int    # exclusive index
    unit_index: int   # Original position in the trajectory (0=O0, 1=A0, 2=O1, ...)

@dataclass
class TokenizedTrajectory:
    """A fully tokenized trajectory, ready for masking and collation."""
    trajectory_id: str
    env: str
    input_ids: torch.LongTensor      # Shape: [seq_len]
    unit_spans: List[TokenizedUnit]  # Metadata for masking

def tokenize_trajectory(
    trajectory: Trajectory,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Optional[TokenizedTrajectory]:
    """
    Tokenizes a Trajectory using the model's chat template to ensure role markers
    and special tokens are correctly included. This is CRITICAL for consistency
    between training and evaluation.
    """
    messages = []
    for unit in trajectory.units:
        role = "user" if unit.unit_type == "obs" else "assistant"
        messages.append({"role": role, "content": unit.text})

    # Use apply_chat_template to get the full tokenized sequence
    # add_generation_prompt=False because we are tokenizing a completed history
    try:
        all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    except Exception:
        # Fallback for older tokenizers or missing templates
        return None

    if len(all_ids) > max_length:
        # If too long, we need to truncate by dropping whole units from the end
        # This is complex because chat templates aren't easily divisible.
        # We iteratively drop units until it fits.
        while len(messages) > 1:
            messages.pop()
            all_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            if len(all_ids) <= max_length:
                break
        else:
            # Even one message is too long
            return None

    # Now we need to identify the spans for each unit.
    # We do this by tokenizing prefixes and finding where they differ.
    unit_spans = []
    current_ids = []
    
    # The LLaDA template adds a SYSTEM prompt at the beginning.
    # We first find the base length with 0 user messages if possible.
    try:
        base_ids = tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=False)
    except:
        base_ids = []
    
    current_pos = len(base_ids)
    
    for i in range(1, len(messages) + 1):
        prefix_ids = tokenizer.apply_chat_template(messages[:i], tokenize=True, add_generation_prompt=False)
        
        # The span starts at the end of the previous prefix and ends at the end of this one.
        # Note: This is an approximation as some templates might add tokens in the middle,
        # but for LLaDA/Llama templates, the new message is appended.
        start_idx = current_pos
        end_idx = len(prefix_ids)
        
        if end_idx > start_idx:
            unit_spans.append(TokenizedUnit(
                unit_type=trajectory.units[i-1].unit_type,
                token_start=start_idx,
                token_end=end_idx,
                unit_index=i-1
            ))
            current_pos = end_idx

    final_input_ids = torch.tensor(all_ids, dtype=torch.long)

    return TokenizedTrajectory(
        trajectory_id=trajectory.id,
        env=trajectory.env,
        input_ids=final_input_ids,
        unit_spans=unit_spans
    )
