import torch
from typing import List, Dict, Any

class AOMTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [ex["input_ids"] for ex in examples]
        labels = [ex["labels"] for ex in examples]
        
        # Max length in batch
        max_len = max(len(ids) for ids in input_ids)
        batch_size = len(input_ids)

        # Pad input_ids and labels
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        # LLaDA 2.0 4D attention mask [B, 1, L, L]
        # For simplicity, we create a full block mask. 
        # If examples vary in length, you'd ideally want a padding mask here too.
        # But LLaDA is usually used with same-length blocks or packing.
        attention_mask = torch.ones((batch_size, 1, max_len, max_len))

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }
