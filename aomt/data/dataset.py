import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
import numpy as np
import os
import pickle
from .tokenize_trajectory import tokenize_trajectory, TokenizedTrajectory, UnitSpan, _ensure_list_of_ints
from .masking import (
    apply_unit_mask,
    sample_sft_mask,
    sample_prefix_stage1_mask,
    sample_aomt_mixed_mask
)

class AOMTDataset(Dataset):
    """
    Unified dataset for all 4 AOMT training methods.
    """
    def __init__(
        self,
        raw_dataset: List[Dict],
        tokenizer,
        method: str,
        p_mask: float = 0.25,
        max_seq_len: int = 2048,
        cache_dir: str = "data/cache",
        split: str = "train",
        model_id: str = "llada2-mini",
        base_seed: int = 42,
        token_level: bool = False
    ):
        self.tokenizer = tokenizer
        self.method = method
        self.p_mask = p_mask
        self.max_seq_len = max_seq_len
        self.base_seed = base_seed
        self._epoch = 0
        self.mask_token_id = tokenizer.mask_token_id
        self.token_level = token_level

        cache_path = os.path.join(cache_dir, f"tokenized_{split}_{model_id.replace('/', '_')}.pkl")
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.tokenized_trajectories = pickle.load(f)
        else:
            self.tokenized_trajectories = []
            for ex in raw_dataset:
                traj = tokenize_trajectory(ex, tokenizer, max_seq_len=max_seq_len)
                if traj and traj.trajectory_length > 0:
                    self.tokenized_trajectories.append(traj)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.tokenized_trajectories, f)

        if method in ["standard_sft", "prefix_sft_stage1", "prefix_sft_stage2"]:
            self.examples = []
            for traj_idx, traj in enumerate(self.tokenized_trajectories):
                for t in range(traj.trajectory_length):
                    self.examples.append((traj_idx, t))
        else:
            self.examples = list(range(len(self.tokenized_trajectories)))

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.method in ["standard_sft", "prefix_sft_stage1", "prefix_sft_stage2"]:
            traj_idx, t = self.examples[idx]
            traj = self.tokenized_trajectories[traj_idx]
            
            if self.method in ["standard_sft", "prefix_sft_stage2"]:
                mask_idx = sample_sft_mask(traj.unit_spans, t)[0]
                end_token = traj.unit_spans[mask_idx].end
                input_ids_full = traj.token_ids[:end_token]
                unit_spans_sliced = traj.unit_spans[:mask_idx+1]
                input_ids, labels = apply_unit_mask(input_ids_full, unit_spans_sliced, [mask_idx], self.mask_token_id)
            else: # prefix_sft_stage1
                mask_idx = sample_prefix_stage1_mask(traj.unit_spans, t)[0]
                ot_idx = mask_idx - 2
                start_token = traj.unit_spans[ot_idx].start
                end_token = traj.unit_spans[mask_idx].end
                input_ids_full = traj.token_ids[start_token:end_token]
                unit_spans_sliced = [
                    UnitSpan(s.start - start_token, s.end - start_token, s.unit_type, s.step_idx)
                    for s in traj.unit_spans[ot_idx : mask_idx + 1]
                ]
                input_ids, labels = apply_unit_mask(input_ids_full, unit_spans_sliced, [2], self.mask_token_id)
        else: # aomt_mixed
            traj = self.tokenized_trajectories[idx]
            rng = np.random.default_rng(self.base_seed ^ idx ^ self._epoch)
            masked_indices = sample_aomt_mixed_mask(
                traj.unit_spans, self.p_mask, rng, 
                token_level=self.token_level, 
                token_ids=traj.token_ids
            )
            input_ids, labels = apply_unit_mask(traj.token_ids, traj.unit_spans, masked_indices, self.mask_token_id)

        # LLaDA 2.0 expects 4D block attention mask [1, L, L] for single example
        # (Trainer will collate this into [B, 1, L, L])
        seq_len = len(input_ids)
        attention_mask = torch.ones((1, seq_len, seq_len), dtype=torch.long)

        # Debug: Check for strings
        for i, val in enumerate(input_ids):
            if not isinstance(val, int):
                print(f"DEBUG: Found non-int at input_ids[{i}]: {val} (type: {type(val)})")
                break

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": attention_mask,
            "method": self.method,
            "unit_span_starts": torch.tensor([s.start for s in traj.unit_spans]),
            "unit_span_ends": torch.tensor([s.end for s in traj.unit_spans])
        }
