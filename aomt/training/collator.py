# aomt/training/collator.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizer

from aomt.data.unit_parser import TokenizedTrajectory

# --- Section 6.2: Prefix SFT Stage 1 Collation ---

def build_prefix_sft_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs multiple short training examples from a single trajectory,
    specifically for the Prefix-SFT (ALEE-style) Stage 1 objective.

    Each example has the form: (O_t, A_t) -> O_{t+1}
    Input:  [O_t tokens] [SEP] [A_t tokens] [SEP] [MASK]...[MASK]
    Target: [O_t tokens] [SEP] [A_t tokens] [SEP] [O_{t+1} tokens]

    Args:
        tokenized_traj (TokenizedTrajectory): The trajectory to process.
        tokenizer (PreTrainedTokenizer): The tokenizer, needed for special tokens.

    Returns:
        List[Dict[str, Any]]: A list of dictionary-based training instances.
    """
    sep_token_id = tokenizer.eos_token_id
    mask_token_id = tokenizer.mask_token_id
    if sep_token_id is None or mask_token_id is None:
        raise ValueError("Tokenizer must have defined eos_token_id and mask_token_id.")

    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []

    # Iterate through the trajectory to find (obs, act, obs) sequences
    for i in range(len(units) - 2):
        # Check for the O_t, A_t, O_{t+1} pattern
        if (units[i].unit_type == "obs" and
            units[i+1].unit_type == "act" and
            units[i+2].unit_type == "obs"):

            obs_t, act_t, obs_t1 = units[i], units[i+1], units[i+2]

            # Context is O_t + SEP + A_t + SEP
            ctx_ids = torch.cat([
                ids[obs_t.token_start:obs_t.token_end],
                torch.tensor([sep_token_id], dtype=torch.long),
                ids[act_t.token_start:act_t.token_end],
                torch.tensor([sep_token_id], dtype=torch.long),
            ])

            # Target is the O_{t+1} span
            target_span = ids[obs_t1.token_start:obs_t1.token_end]
            
            # The input replaces the target span with MASK tokens of the same length
            masked_span = torch.full_like(target_span, fill_value=mask_token_id)
            
            # Assemble the full sequences
            full_input_ids = torch.cat([ctx_ids, masked_span])
            # The target_ids contain the ground truth for the whole sequence
            full_target_ids = torch.cat([ctx_ids, target_span])
            
            # Loss is only calculated on the O_{t+1} span
            loss_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
            loss_mask[len(ctx_ids):] = True

            examples.append({
                "input_ids": full_input_ids,
                "target_ids": full_target_ids,
                "loss_mask": loss_mask,
                "use_causal_mask": True, # User-directed change to make Stage 1 causal
            })

    return examples


# --- Section 6.3: Prefix SFT Stage 2 Collation ---

def build_prefix_sft_stage2_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs multiple short training examples from a single trajectory
    for the Prefix-SFT Stage 2 (Policy) objective.

    Each example has the form: O_t -> A_t
    Input:  [O_t tokens] [SEP] [MASK]...[MASK]
    Target: [O_t tokens] [SEP] [A_t tokens]

    Args:
        tokenized_traj (TokenizedTrajectory): The trajectory to process.
        tokenizer (PreTrainedTokenizer): The tokenizer, needed for special tokens.

    Returns:
        List[Dict[str, Any]]: A list of dictionary-based training instances.
    """
    sep_token_id = tokenizer.eos_token_id
    mask_token_id = tokenizer.mask_token_id
    if sep_token_id is None or mask_token_id is None:
        raise ValueError("Tokenizer must have defined eos_token_id and mask_token_id.")

    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []

    # Iterate through the trajectory to find (obs, act) sequences
    for i in range(len(units) - 1):
        if units[i].unit_type == "obs" and units[i+1].unit_type == "act":
            obs_t, act_t = units[i], units[i+1]

            # Context is O_t + SEP
            ctx_ids = torch.cat([
                ids[obs_t.token_start:obs_t.token_end],
                torch.tensor([sep_token_id], dtype=torch.long),
            ])

            # Target is the A_t span
            target_span = ids[act_t.token_start:act_t.token_end]
            
            # The input replaces the target span with MASK tokens
            masked_span = torch.full_like(target_span, fill_value=mask_token_id)
            
            # Assemble the full sequences
            full_input_ids = torch.cat([ctx_ids, masked_span])
            full_target_ids = torch.cat([ctx_ids, target_span])
            
            # Loss is only calculated on the A_t span
            loss_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
            loss_mask[len(ctx_ids):] = True

            examples.append({
                "input_ids": full_input_ids,
                "target_ids": full_target_ids,
                "loss_mask": loss_mask,
                "use_causal_mask": True,
            })

    return examples


def build_standard_sft_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs multiple short training examples from a single trajectory to
    efficiently mimic autoregressive SFT with a bidirectional model.

    For each action `a_t` in the trajectory, it creates an example:
    (s_0, a_0, ..., s_t) -> a_t

    Input:  [s_0, a_0, ..., s_t tokens] [SEP] [MASK]...[MASK]
    Target: [s_0, a_0, ..., s_t tokens] [SEP] [a_t tokens]

    Args:
        tokenized_traj (TokenizedTrajectory): The trajectory to process.
        tokenizer (PreTrainedTokenizer): The tokenizer, needed for special tokens.

    Returns:
        List[Dict[str, Any]]: A list of dictionary-based training instances.
    """
    sep_token_id = tokenizer.eos_token_id
    mask_token_id = tokenizer.mask_token_id
    if sep_token_id is None or mask_token_id is None:
        raise ValueError("Tokenizer must have defined eos_token_id and mask_token_id.")

    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []

    # Iterate through the trajectory to find all actions
    for i in range(len(units)):
        if units[i].unit_type == "act":
            act_t = units[i]
            
            # Context is everything up to the start of the current action
            # This includes all prior obs and act units.
            # The final observation before the action is units[i-1].
            if i == 0:
                continue # Cannot predict an action with no preceding observation
            
            prev_unit_end = units[i-1].token_end
            
            ctx_ids = torch.cat([
                ids[:prev_unit_end],
                torch.tensor([sep_token_id], dtype=torch.long),
            ])

            # Target is the A_t span
            target_span = ids[act_t.token_start:act_t.token_end]
            
            # The input replaces the target span with MASK tokens
            masked_span = torch.full_like(target_span, fill_value=mask_token_id)
            
            # Assemble the full sequences
            full_input_ids = torch.cat([ctx_ids, masked_span])
            full_target_ids = torch.cat([ctx_ids, target_span])
            
            # Loss is only calculated on the A_t span
            loss_mask = torch.zeros_like(full_input_ids, dtype=torch.bool)
            loss_mask[len(ctx_ids):] = True

            examples.append({
                "input_ids": full_input_ids,
                "target_ids": full_target_ids,
                "loss_mask": loss_mask,
                "use_causal_mask": True,
            })
            
    return examples


# --- Standard Data Collator for AOMT and SFT ---

@dataclass
class AOMTDataCollator:
    """
    Pads batches of tokenized data to the maximum length in each batch.
    
    This is used for all training modes except for the specialized 
    PREFIX_SFT_STAGE1, which requires the `build_prefix_sft_examples` logic
    to be applied first.
    """
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Determine padding token ID
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Find the max length in the current batch
        max_len = max(len(f["input_ids"]) for f in features)

        # Initialize lists to hold padded tensors
        padded_input_ids = []
        padded_target_ids = []
        padded_loss_masks = []
        attention_masks = []

        for feature in features:
            input_ids = feature["input_ids"]
            remainder = max_len - len(input_ids)

            # Pad input_ids, creating the attention mask simultaneously
            padded_input_ids.append(torch.nn.functional.pad(
                input_ids, (0, remainder), value=pad_token_id
            ))
            attention_masks.append(torch.cat([
                torch.ones(len(input_ids)), torch.zeros(remainder)
            ]))
            
            # Pad target_ids (ground truth)
            padded_target_ids.append(torch.nn.functional.pad(
                feature["target_ids"], (0, remainder), value=pad_token_id
            ))
            
            # Pad loss_mask with False (0)
            padded_loss_masks.append(torch.nn.functional.pad(
                feature["loss_mask"], (0, remainder), value=False
            ))
        
        # Stack all features into a single batch tensor
        batch = {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks).long(),
            "target_ids": torch.stack(padded_target_ids),
            "loss_mask": torch.stack(padded_loss_masks),
            # Propagate the causal mask flag. This assumes it's the same for all
            # features in the batch, which is a safe assumption as it's
            # determined by the dataset mode.
            "use_causal_mask": torch.tensor([f["use_causal_mask"] for f in features]),
        }
        return batch
