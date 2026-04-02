# aomt/training/objectives.py
import torch
import torch.nn.functional as F

def masked_unit_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor
) -> torch.Tensor:
    """
    Calculates cross-entropy loss only on tokens specified by the loss_mask.

    This is the central loss function for all AOMT training modes. It ensures
    that only the tokens of the designated "target" units (which may have been
    masked in the input) contribute to the loss.

    Args:
        logits (torch.Tensor): The model's output logits.
                               Shape: [batch_size, seq_len, vocab_size]
        target_ids (torch.Tensor): The original, clean token IDs (the ground truth).
                                   Shape: [batch_size, seq_len]
        loss_mask (torch.Tensor): A boolean mask where `True` indicates a token
                                  position that should be included in the loss.
                                  Shape: [batch_size, seq_len]

    Returns:
        torch.Tensor: A scalar tensor representing the mean loss per masked token.
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross_entropy:
    # Logits: [B, S, V] -> [B * S, V]
    # Target: [B, S] -> [B * S]
    # loss_mask: [B, S] -> [B * S]
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = target_ids.view(-1)
    loss_mask_flat = loss_mask.view(-1)

    # Calculate cross-entropy for all tokens, but don't reduce yet
    loss_all = F.cross_entropy(logits_flat, targets_flat, reduction="none")

    # Apply the mask to the loss
    loss_masked = loss_all * loss_mask_flat.float()

    # Normalize the loss by the number of masked tokens.
    # .clamp(min=1.0) prevents division by zero if a batch has no masked tokens.
    num_masked_tokens = loss_mask_flat.float().sum()
    mean_loss = loss_masked.sum() / num_masked_tokens.clamp(min=1.0)

    return mean_loss
