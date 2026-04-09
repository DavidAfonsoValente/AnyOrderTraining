import pytest
import torch
from omegaconf import OmegaConf
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.training.losses import masked_cross_entropy_loss
from transformers import AutoTokenizer

@pytest.fixture
def mock_config():
    return OmegaConf.create({
        "lr": 2.5e-5,
        "weight_decay": 0.1,
        "batch_size": 2,
        "epochs": 1,
        "lr_schedule": "cosine",
        "warmup_steps": 0,
        "grad_clip": 1.0,
        "checkpoint_save_steps": 100,
        "output_dir": "test_output",
        "gradient_accumulation_steps": 1,
        "use_wandb": False
    })

def test_loss_backward():
    # Smoke test for loss and grad_fn
    logits = torch.randn(2, 10, 100, requires_grad=True)
    labels = torch.full((2, 10), -100, dtype=torch.long)
    labels[0, 2:5] = torch.tensor([10, 20, 30])
    
    loss = masked_cross_entropy_loss(logits, labels)
    assert loss.item() > 0
    assert loss.grad_fn is not None
    
    loss.backward()
    assert logits.grad is not None
    assert (logits.grad != 0).any()

def test_dataset_mask_diversity(synthetic_trajectory, tokenizer_fixture):
    raw_dataset = [synthetic_trajectory]
    ds = AOMTDataset(raw_dataset, tokenizer_fixture, method="aomt_mixed", p_mask=0.5)
    
    ds.set_epoch(0)
    item1 = ds[0]
    
    ds.set_epoch(1)
    item2 = ds[0]
    
    # Masks should differ between epochs with high probability
    assert not torch.equal(item1["input_ids"], item2["input_ids"])

def test_collator_padding(tokenizer_fixture):
    collator = AOMTDataCollator(tokenizer_fixture)
    examples = [
        {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1])},
        {"input_ids": torch.tensor([4, 5]), "labels": torch.tensor([4, 5]), "attention_mask": torch.tensor([1, 1])}
    ]
    batch = collator(examples)
    
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["input_ids"][1, 2] == (tokenizer_fixture.pad_token_id or tokenizer_fixture.eos_token_id)
    assert batch["labels"][1, 2] == -100
    assert batch["attention_mask"][1, 2] == 0
