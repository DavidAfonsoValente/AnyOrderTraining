import torch
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Tuple

class TrajectoryTransform:
    """
    Processes a raw trajectory of alternating observations and actions into a
    tokenized sequence, while tracking the boundaries of each unit.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initializes the transform with a tokenizer.
        
        Args:
            tokenizer: A Hugging Face tokenizer.
        """
        self.tokenizer = tokenizer

    def process_trajectory(self, trajectory: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Tokenizes a trajectory and identifies the boundaries of each unit.

        Args:
            trajectory: A list of dictionaries, where each dict has a single
                        key "observation" or "action" and a string value.
                        Example: [{'observation': '...'}, {'action': '...'}, ...]

        Returns:
            A dictionary containing:
            - 'input_ids': A torch.Tensor of the tokenized trajectory.
            - 'attention_mask': A torch.Tensor of the attention mask.
            - 'unit_boundaries': A list of tuples, where each tuple is the
                                 [start, end) token indices of a unit.
            - 'unit_types': A list of strings ('observation' or 'action')
                            corresponding to each unit boundary.
        """
        all_input_ids = []
        unit_boundaries = []
        unit_types = []

        current_pos = 0
        
        # Add BOS token at the beginning if the tokenizer uses one
        if self.tokenizer.bos_token_id is not None:
            all_input_ids.append(self.tokenizer.bos_token_id)
            current_pos += 1

        for unit in trajectory:
            if not unit:
                continue
            unit_type, unit_text = list(unit.items())[0]
            
            # Add a separator and a prefix to give the model a clear structural signal
            formatted_text = f"
{unit_type.capitalize()}: {unit_text}"
            
            # Tokenize the unit text without adding special tokens, as we're managing them manually
            unit_ids = self.tokenizer(formatted_text, add_special_tokens=False).input_ids
            
            if not unit_ids:
                continue

            start_idx = current_pos
            end_idx = start_idx + len(unit_ids)
            
            all_input_ids.extend(unit_ids)
            unit_boundaries.append((start_idx, end_idx))
            unit_types.append(unit_type)
            
            current_pos = end_idx

        # Add EOS token at the end if the tokenizer uses one
        if self.tokenizer.eos_token_id is not None:
            all_input_ids.append(self.tokenizer.eos_token_id)

        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids_tensor)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "unit_boundaries": unit_boundaries,
            "unit_types": unit_types,
        }

def get_unit_indices(processed_trajectory: Dict[str, Any], unit_type_to_get: str) -> List[int]:
    """
    Helper function to get the indices of specific unit types from a processed trajectory.
    
    Args:
        processed_trajectory: The output of TrajectoryTransform.process_trajectory.
        unit_type_to_get: The type of unit to get indices for ('observation' or 'action').
        
    Returns:
        A list of integer indices corresponding to the desired unit type.
    """
    return [i for i, unit_type in enumerate(processed_trajectory['unit_types']) if unit_type == unit_type_to_get]
