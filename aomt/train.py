import argparse
import os
from omegaconf import OmegaConf
from datasets import load_dataset
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.training.trainer import AOMTTrainer
from aomt.data.utils import load_robust_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["standard_sft", "prefix_sft_stage1", "prefix_sft_stage2", "aomt_mixed"])
    parser.add_argument("--p_mask", type=float)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--init_checkpoint", type=str, help="Initialize from this checkpoint instead of base model")
    args = parser.parse_args()

    # Load base config and merge with specific config
    base_cfg = OmegaConf.load("aomt/config/base.yaml")
    run_cfg = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_cfg, run_cfg)
    
    # Overrides
    if args.method: config.method = args.method
    if args.p_mask: config.p_mask = args.p_mask
    if args.output_dir: config.output_dir = args.output_dir
    if args.init_checkpoint: config.model_id = args.init_checkpoint
    
    print(f"Starting training for method: {config.method}")
    
    # Load dataset
    print("Loading dataset...")
    # Use the robust loader we explored in Phase 1
    ds_dict, _ = load_robust_dataset(config.dataset_name)
    train_raw = ds_dict["train"]
    
    # Load model and tokenizer
    print(f"Loading model and tokenizer from {config.model_id}...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=config.model_id,
        precision=config.precision
    )
    
    # Create dataset
    train_dataset = AOMTDataset(
        raw_dataset=train_raw,
        tokenizer=tokenizer,
        method=config.method,
        p_mask=config.get("p_mask", 0.25),
        max_seq_len=config.max_seq_len,
        split="train",
        model_id=config.model_id
    )
    
    # Create collator
    collator = AOMTDataCollator(tokenizer)
    config.collator_fn = collator
    
    # Initialize trainer
    trainer = AOMTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
        method=config.method
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()
