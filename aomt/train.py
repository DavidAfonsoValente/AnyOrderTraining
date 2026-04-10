import os
import torch
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from aomt.model.llada_wrapper import load_model_and_tokenizer
from aomt.data.dataset import AOMTDataset
from aomt.data.collator import AOMTDataCollator
from aomt.data.utils import load_robust_dataset
from aomt.training.losses import masked_cross_entropy_loss
import argparse
from omegaconf import OmegaConf

class AOMTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        attention_mask = inputs["attention_mask"].to(model.dtype)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask
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
    
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available.")
    
    # 1. Load Base Model
    model, tokenizer = load_model_and_tokenizer(
        model_id="aomt/weights/LLaDA2.0-mini",
        precision="bf16",
        device_map="auto"
    )

    # 2. Apply LoRA (The "Memory Savior")
    print("Applying LoRA for memory efficiency...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value", "dense", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Data Setup
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

    # 4. Training Args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 64),
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        bf16=True,
        gradient_checkpointing=True,
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
    # Save the LoRA adapter
    trainer.save_model(args.output_dir)
    print(f"Training complete. Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
