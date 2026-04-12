import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Monkey-patch nn.Module to ensure set_submodule is available for bitsandbytes
if not hasattr(nn.Module, "set_submodule"):
    def set_submodule(self, target: str, module: nn.Module) -> None:
        if not target:
            raise ValueError("Target path cannot be empty.")
        parts = target.split(".")
        target_mod = self.get_submodule(".".join(parts[:-1]))
        setattr(target_mod, parts[-1], module)
    nn.Module.set_submodule = set_submodule
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
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        labels = inputs["labels"]
        loss = masked_cross_entropy_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Path to previous stage checkpoint")
    args = parser.parse_args()

    base_config_path = os.path.join(os.path.dirname(__file__), "config/base.yaml")
    config = OmegaConf.load(base_config_path)
    exp_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(config, exp_config)
    
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CUDA is not available.")
    
    # 1. Configure 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. Load Model with Quantization
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # We load directly here to ensure bnb_config is applied
    tokenizer = AutoTokenizer.from_pretrained("aomt/weights/LLaDA2.0-mini", trust_remote_code=True)
    
    model_path = args.init_checkpoint if args.init_checkpoint else "aomt/weights/LLaDA2.0-mini"
    print(f"Loading model from: {model_path} in 4-bit (QLoRA)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. Prepare for LoRA
    if not isinstance(model, PeftModel):
        print("Initializing new LoRA adapter...")
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query_key_value", "dense", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Detected existing PEFT model, skipping re-initialization.")
    
    model.print_trainable_parameters()

    # 4. Data Setup
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

    # 5. Training Args (Using Paged Optimizer)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 64),
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit", # Memory-efficient paged optimizer
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
    print(f"Training complete. QLoRA Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
