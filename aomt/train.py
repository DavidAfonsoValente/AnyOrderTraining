import os
import torch
from transformers import Trainer, TrainingArguments
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.data.utils import load_robust_dataset
from aomt.training.losses import masked_cross_entropy_loss
import argparse
from omegaconf import OmegaConf

class AOMTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Our collator already provides the 4D mask LLaDA 2.0 requires
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        logits = outputs.logits
        labels = inputs["labels"]
        loss = masked_cross_entropy_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Configuration setup
    base_config_path = os.path.join(os.path.dirname(__file__), "config/base.yaml")
    config = OmegaConf.load(base_config_path)
    exp_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, exp_config)
    
    print(f"Starting training for method: {config.method}")
    
    # 1. Hardware Detection
    has_cuda = torch.cuda.is_available()
    # On some cluster nodes, is_bf16_supported() returns False even if H100 is present
    # due to driver/toolkit version mismatches.
    bf16_available = has_cuda and torch.cuda.is_bf16_supported()
    fp16_available = has_cuda and not bf16_available
    
    print(f"CUDA Available: {has_cuda}")
    print(f"BF16 Supported: {bf16_available}")
    
    # 2. Model Loading (Model itself stays in BF16 if possible)
    model, tokenizer = load_model_and_tokenizer(
        model_id="aomt/weights/LLaDA2.0-mini",
        precision="bf16" if (bf16_available or has_cuda) else "fp32",
        device_map="auto"
    )

    # 3. Dataset & Collator
    raw_datasets, _ = load_robust_dataset()
    train_dataset = AOMTDataset(
        raw_datasets["train"],
        tokenizer,
        method=config.method,
        p_mask=config.p_mask,
        max_seq_len=config.max_seq_len,
        token_level=config.get("token_level", False)
    )
    collator = AOMTDataCollator(tokenizer)

    # 4. Training Arguments
    # We only set the precision flags that the environment explicitly supports
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        bf16=bf16_available,
        fp16=fp16_available,
        logging_steps=10,
        save_steps=config.get("checkpoint_save_steps", 500),
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = AOMTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
