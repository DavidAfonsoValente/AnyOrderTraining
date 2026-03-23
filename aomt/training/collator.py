# aomt/training/collator.py
"""
Data collation utilities for AOMT training.

NOTE: The primary training pipeline uses train_aomt.py and train_standard_sft.py,
which handle data loading/collation internally. The functions below are utility
functions for data preprocessing (prepare_data.py) and are NOT called during
training.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from transformers import PreTrainedTokenizer

from aomt.data.unit_parser import TokenizedTrajectory


def build_prefix_sft_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs Prefix-SFT Stage 1 examples: (O_t, A_t) -> O_{t+1}.
    
    Paper (Sec. 4, Appendix A):
        "Stage 1 predicts O_{t+1} from local context {O_t, A_t}."
        "each datapoint contains only {O_t, A_t} as prompt and O_{t+1} as response."
    
    Returns examples in chat format (messages list) for compatibility with
    train_standard_sft.py's SFTDataset.
    """
    units = tokenized_traj.unit_spans
    ids = tokenized_traj.input_ids
    examples = []

    for i in range(len(units) - 2):
        if (units[i].unit_type == "obs" and
            units[i+1].unit_type == "act" and
            units[i+2].unit_type == "obs"):

            obs_t = ids[units[i].token_start:units[i].token_end]
            act_t = ids[units[i+1].token_start:units[i+1].token_end]
            obs_t1 = ids[units[i+2].token_start:units[i+2].token_end]

            # Decode to text for chat template format
            obs_t_text = tokenizer.decode(obs_t, skip_special_tokens=True)
            act_t_text = tokenizer.decode(act_t, skip_special_tokens=True)
            obs_t1_text = tokenizer.decode(obs_t1, skip_special_tokens=True)

            # Local context {O_t, A_t} as prompt, O_{t+1} as response
            messages = [
                {"role": "user", "content": f"{obs_t_text}\n{act_t_text}"},
                {"role": "assistant", "content": obs_t1_text}
            ]

            examples.append({"messages": messages})

    return examples


def build_standard_sft_examples(
    tokenized_traj: TokenizedTrajectory,
    tokenizer: PreTrainedTokenizer
) -> List[Dict[str, Any]]:
    """
    Constructs Standard SFT examples: full causal prefix -> A_t.
    
    Paper (Sec. 4):
        "slices each trajectory into T examples, one per action; each example's
        causal prefix is the prompt and the action A_t is the fully-masked response."
    
    Also used for Prefix SFT Stage 2 (paper: "Stage 2 fine-tunes from the
    Stage 1 checkpoint using the same data and masking as Standard SFT").
    
    Returns examples in chat format (messages list) for compatibility with
    train_standard_sft.py's SFTDataset.
    """
    units = tokenized_traj.unit_spans
    examples = []

    for i in range(len(units)):
        if units[i].unit_type != "act":
            continue
        if i == 0:
            continue  # No preceding observation

        # Build message history: all units up to and including the current action
        messages = []
        for j in range(i + 1):
            role = "user" if units[j].unit_type == "obs" else "assistant"
            text = tokenizer.decode(
                tokenized_traj.input_ids[units[j].token_start:units[j].token_end],
                skip_special_tokens=True
            )
            messages.append({"role": role, "content": text})

        # The prompt is everything up to the last message (the action)
        # The response is the action itself
        # train_standard_sft.py will split prompt/response using apply_chat_template
        examples.append({"messages": messages})

    return examples


# --- Standard Data Collator for AOMT ---

@dataclass
class AOMTDataCollator:
    """
    Pads batches of tokenized data to the maximum length in each batch.
    Uses bidirectional attention (no causal mask) per the paper.
    
    Paper (Sec. 3.2): "p_theta attends bidirectionally over all context units."
    """
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        max_len = max(len(f["input_ids"]) for f in features)

        padded_input_ids = []
        padded_target_ids = []
        padded_loss_masks = []
        attention_masks = []

        for feature in features:
            input_ids = feature["input_ids"]
            remainder = max_len - len(input_ids)

            padded_input_ids.append(torch.nn.functional.pad(
                input_ids, (0, remainder), value=pad_token_id
            ))
            attention_masks.append(torch.cat([
                torch.ones(len(input_ids)), torch.zeros(remainder)
            ]))

            padded_target_ids.append(torch.nn.functional.pad(
                feature["target_ids"], (0, remainder), value=pad_token_id
            ))

            padded_loss_masks.append(torch.nn.functional.pad(
                feature["loss_mask"], (0, remainder), value=False
            ))

        batch = {
            "input_ids": torch.stack(padded_input_ids),
            # Bidirectional attention mask (paper: no causal mask)
            "attention_mask": torch.stack(attention_masks).long(),
            "target_ids": torch.stack(padded_target_ids),
            "loss_mask": torch.stack(padded_loss_masks),
        }
        return batch
