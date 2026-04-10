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
        # LLaDA 2.0 requires 4D mask
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

    base_config_path = os.path.join(os.path.dirname(__file__), "config/base.yaml")
    config = OmegaConf.load(base_config_path)
    exp_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, exp_config)
    
    # HARD GPU CHECK
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available. Training on CPU is disabled to prevent node hang.")
    
    device_name = torch.cuda.get_device_name(0)
    print(f"### Starting AOMT Training on GPU: {device_name} ###")
    print(f"### Method: {config.method} | BF16: {torch.cuda.is_bf16_supported()} ###")
    
    model, tokenizer = load_model_and_tokenizer(
        model_id="aomt/weights/LLaDA2.0-mini",
        precision="bf16", # We force bf16 because we know H100 supports it
        device_map="auto"
    )

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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        bf16=True, # H100 always supports this
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
