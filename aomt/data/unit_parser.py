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
    Parses a raw conversation from the dataset into a structured Trajectory object.

    Args:
        example (dict): A single example from the Hugging Face dataset.

    Returns:
        Trajectory: A structured representation of the conversation.
    """
    units = []
    for i, turn in enumerate(example["conversations"]):
        utype = "obs" if turn["from"] == "human" else "act"
        # The first 'human' turn is O_0 and can contain task instructions.
        units.append(TrajectoryUnit(unit_type=utype, text=turn["value"], turn_index=i))

    # Validate that the turns strictly alternate between "obs" and "act"
    for i, u in enumerate(units):
        expected_utype = "obs" if i % 2 == 0 else "act"
        if u.unit_type != expected_utype:
            raise ValueError(f"Trajectory {example['id']}, Turn {i}: "
                             f"Expected unit type '{expected_utype}' but got '{u.unit_type}'.")

    return Trajectory(id=example["id"], env=example.get("env", example.get("task", "unknown")), units=units)


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
    Tokenizes a Trajectory into a flat sequence and records unit spans.

    The format is: [O0_tokens] [SEP] [A0_tokens] [SEP] [O1_tokens] ...

    A separator token is inserted between units. If the total length exceeds
    max_length, whole units are truncated from the END of the trajectory.

    Args:
        trajectory (Trajectory): The structured trajectory to tokenize.
        tokenizer (PreTrainedTokenizer): The tokenizer to use.
        max_length (int): The maximum sequence length for the tokenized output.

    Returns:
        Optional[TokenizedTrajectory]: The tokenized trajectory, or None if it's empty after truncation.
    """
    # Use eos_token as the separator for LLaMA-style models.
    sep_token_id = tokenizer.eos_token_id
    if sep_token_id is None:
        raise ValueError("Tokenizer must have a defined `eos_token_id` to be used as a separator.")

    all_ids_list = []
    unit_spans: List[TokenizedUnit] = []
    current_pos = 0

    for i, unit in enumerate(trajectory.units):
        unit_ids = tokenizer.encode(unit.text, add_special_tokens=False)
        if not unit_ids:
            continue # Skip empty units

        # Check if adding this unit would exceed max_length
        # Add 1 for the separator token if it's not the last unit
        projected_len = current_pos + len(unit_ids) + (1 if i < len(trajectory.units) - 1 else 0)
        if projected_len > max_length:
            break # Stop processing, effectively truncating from this unit onwards

        # Add the unit's tokens
        all_ids_list.extend(unit_ids)
        
        start_idx = current_pos
        end_idx = start_idx + len(unit_ids)
        current_pos = end_idx

        unit_spans.append(TokenizedUnit(
            unit_type=unit.unit_type,
            token_start=start_idx,
            token_end=end_idx,
            unit_index=i
        ))

        # Add a separator token if it's not the last unit
        if i < len(trajectory.units) - 1:
            # Check if there is space for the separator
            if current_pos < max_length:
                all_ids_list.append(sep_token_id)
                current_pos += 1
            else:
                # No space for the separator, so the last unit is effectively the last one
                break
    
    if not unit_spans:
        return None

    # Final tensor of token IDs
    final_input_ids = torch.tensor(all_ids_list, dtype=torch.long)

    return TokenizedTrajectory(
        trajectory_id=trajectory.id,
        env=trajectory.env,
        input_ids=final_input_ids,
        unit_spans=unit_spans
    )
