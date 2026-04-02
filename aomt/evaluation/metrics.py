import torch
import torch.nn.functional as F
from typing import List
from ..data.tokenize_trajectory import TokenizedTrajectory
from ..data.masking import apply_unit_mask

@torch.no_grad()
def compute_observation_masked_nll(
    model, tokenizer,
    test_trajectories: List[TokenizedTrajectory],
    method: str,
    device: str = "cuda",
) -> float:
    """
    Computes obs-masked NLL as defined in the paper.
    """
    if method in ["standard_sft", "prefix_sft_stage1"]:
        raise ValueError(f"NLL_obs is not applicable for {method}")

    model.eval()
    mask_token_id = tokenizer.mask_token_id
    
    total_log_prob = 0.0
    total_tokens = 0
    
    for traj in test_trajectories:
        # For each observation unit O_t
        obs_indices = [i for i, s in enumerate(traj.unit_spans) if s.unit_type == "observation"]
        
        for obs_idx in obs_indices:
            # Construct input with ONLY this observation masked
            masked_ids, labels = apply_unit_mask(
                traj.token_ids,
                traj.unit_spans,
                [obs_idx],
                mask_token_id
            )
            
            input_tensor = torch.tensor(masked_ids).unsqueeze(0).to(device)
            label_tensor = torch.tensor(labels).unsqueeze(0).to(device)
            
            outputs = model(input_tensor)
            logits = outputs.logits[0] # (seq_len, vocab)
            
            # Mask for current observation tokens
            span = traj.unit_spans[obs_idx]
            obs_logits = logits[span.start:span.end]
            obs_labels = label_tensor[0, span.start:span.end]
            
            # log p(O_t | context) = sum log p(token_i | context)
            log_probs = F.log_softmax(obs_logits, dim=-1)
            # Gather log probs for the ground truth tokens
            # labels are the original token IDs
            token_log_probs = log_probs.gather(1, obs_labels.unsqueeze(1)).squeeze(1)
            
            total_log_prob += token_log_probs.sum().item()
            total_tokens += len(obs_labels)
            
    if total_tokens == 0:
        return 0.0
        
    # NLL = - (sum log prob) / num_observations? 
    # The paper says E_tau E_t [ log p ]. 
    # Usually it's averaged over masked positions.
    return -total_log_prob / total_tokens
