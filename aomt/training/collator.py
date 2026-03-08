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
    Constructs Prefix-SFT Stage 1 examples: (O_t, A_t) -> O_{t+1}.
    Prompt = O_t, A_t (local context ONLY). Response = O_{t+1}.
    """
    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []

    for i in range(len(units) - 2):
        if (units[i].unit_type == "obs" and
            units[i+1].unit_type == "act" and
            units[i+2].unit_type == "obs"):

            obs_t = tokenized_traj.input_ids[units[i].token_start:units[i].token_end]
            act_t = tokenized_traj.input_ids[units[i+1].token_start:units[i+1].token_end]
            obs_t1 = tokenized_traj.input_ids[units[i+2].token_start:units[i+2].token_end]

            # Reconstruct text to use chat template for consistent boundaries
            obs_t_text = tokenizer.decode(obs_t)
            act_t_text = tokenizer.decode(act_t)
            obs_t1_text = tokenizer.decode(obs_t1)

            messages = [
                {"role": "user", "content": f"{obs_t_text}\n{act_t_text}"},
                {"role": "assistant", "content": obs_t1_text}
            ]
            
            # This will be processed by SFTDataset in train_standard_sft.py
            examples.append({"messages": messages})

    return examples


# --- Section 6.3: Prefix SFT Stage 2 Collation ---

def build_prefix_sft_stage2_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs Prefix-SFT Stage 2 (Policy) examples: O_t -> A_t.
    """
    units = tokenized_traj.unit_spans
    examples = []

    for i in range(len(units) - 1):
        if units[i].unit_type == "obs" and units[i+1].unit_type == "act":
            obs_t = tokenized_traj.input_ids[units[i].token_start:units[i].token_end]
            act_t = tokenized_traj.input_ids[units[i+1].token_start:units[i+1].token_end]

            messages = [
                {"role": "user", "content": tokenizer.decode(obs_t)},
                {"role": "assistant", "content": tokenizer.decode(act_t)}
            ]
            examples.append({"messages": messages})

    return examples


def build_standard_sft_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs Standard SFT examples: (O_0, A_0, ..., O_t) -> A_t.
    """
    units = tokenized_traj.unit_spans
    examples = []

    # Iterate through the trajectory to find all actions
    for i in range(len(units)):
        if units[i].unit_type == "act":
            if i == 0: continue
            
            # Context is everything up to the start of the current action
            prompt_ids = tokenized_traj.input_ids[:units[i-1].token_end]
            target_ids = tokenized_traj.input_ids[units[i].token_start:units[i].token_end]

            messages = [
                {"role": "user", "content": tokenizer.decode(prompt_ids)},
                {"role": "assistant", "content": tokenizer.decode(target_ids)}
            ]
            examples.append({"messages": messages})
            
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
