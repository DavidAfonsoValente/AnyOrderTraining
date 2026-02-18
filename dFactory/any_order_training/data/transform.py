
import torch
from typing import  Any, Dict, List, Optional, Sequence, Union, Tuple
from ..sampler import AnyOrderMaskSampler

def process_any_order_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    training_mode: str = 'any_order',
    mask_prob: float = 0.5,
    mask_token: str = "[MASK]",
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    # Create a trajectory from the messages
    trajectory = [message["content"] for message in messages]

    # Apply masking based on the training mode
    if training_mode == 'any_order':
        sampler = AnyOrderMaskSampler(mask_prob=mask_prob, mask_token=mask_token)
        masked_trajectory, labels, _ = sampler(trajectory)
    elif training_mode == 'causal':
        # For causal, we mask the last action
        masked_trajectory = trajectory[:-1] + [mask_token]
        labels = [None] * (len(trajectory) - 1) + [trajectory[-1]]
    elif training_mode == 'prefix':
        # For prefix, we mask everything after the first observation and action
        masked_trajectory = trajectory[:2] + [mask_token] * (len(trajectory) - 2)
        labels = [None, None] + trajectory[2:]
    else:
        raise ValueError(f"Invalid training_mode: {training_mode}")

    # Reconstruct the messages with the masked trajectory
    masked_messages = []
    for i, message in enumerate(messages):
        masked_messages.append({
            "role": message["role"],
            "content": masked_trajectory[i]
        })

    # Tokenize the masked messages
    inputs_str = tokenizer.apply_chat_template(masked_messages, tokenize=False, add_generation_prompt=False)
    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        add_special_tokens=False
    ).input_ids.squeeze(0)

    # Create labels for the masked parts
    labels_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized_labels = tokenizer(
        labels_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        add_special_tokens=False
    ).input_ids.squeeze(0)

    # Only compute loss on the masked parts
    final_labels = tokenized_labels.clone()
    final_labels[tokenized_input == tokenized_labels] = -100


    return [{
        "input_ids": tokenized_input,
        "attention_mask": (tokenized_input != tokenizer.pad_token_id).long(),
        "labels": final_labels,
    }]
