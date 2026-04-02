import torch
import torch.nn.functional as F

def masked_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Computes cross-entropy loss averaged over masked token positions only.
    """
    # logits: (batch, seq_len, vocab_size)
    # labels: (batch, seq_len) with -100 at ignored positions
    
    # Flatten tensors
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)
    
    # Check if there are any non-(-100) positions to avoid NaN
    if not (labels_flat != -100).any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
    return F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
