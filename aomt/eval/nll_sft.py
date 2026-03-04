# aomt/eval/nll_sft.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from aomt.data.unit_parser import TokenizedTrajectory
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def compute_nll_sft(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    tokenized_trajectories: list[TokenizedTrajectory],
    batch_size: int = 4,
    device: str = "cuda"
) -> dict:
    """
    Computes the standard SFT Negative Log-Likelihood on a held-out dataset.

    This is the standard causal language modeling loss, which evaluates the model's
    ability to predict the next token in a sequence.

    Args:
        model: The trained model to evaluate.
        tokenizer: The tokenizer used for training.
        tokenized_trajectories: A list of tokenized trajectories for the eval set.
        batch_size (int): The batch size for evaluation.
        device (str): The device to run evaluation on ("cuda" or "cpu").

    Returns:
        dict: A dictionary containing the mean SFT NLL.
    """
    model.eval()
    model.to(device)

    total_nll = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(tokenized_trajectories), batch_size), desc="Evaluating SFT NLL"):
        batch_trajectories = tokenized_trajectories[i:i+batch_size]
        
        # Prepare batch tensors
        max_len = max(len(traj.input_ids) for traj in batch_trajectories)
        input_ids_batch = []
        attention_mask_batch = []
        
        for traj in batch_trajectories:
            padding_len = max_len - len(traj.input_ids)
            input_ids_batch.append(F.pad(traj.input_ids, (0, padding_len), value=tokenizer.pad_token_id))
            attention_mask_batch.append(F.pad(torch.ones_like(traj.input_ids), (0, padding_len), value=0))

        input_ids_tensor = torch.stack(input_ids_batch).to(device)
        attention_mask_tensor = torch.stack(attention_mask_batch).to(device)

        # Forward pass with causal attention mask
        outputs = model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
        )
        logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids_tensor[..., 1:].contiguous()

        # Calculate loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        total_nll += loss.item()
        total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()

    mean_nll = total_nll / total_tokens if total_tokens > 0 else 0.0
    
    return {"mean_nll_sft": mean_nll}
