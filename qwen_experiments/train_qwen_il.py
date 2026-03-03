import argparse
import logging
import os
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a Qwen model for Causal Language Modeling (IL).")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--train_file", type=str, required=True, help="The input training data file (jsonl).")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to store the final model.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="The initial learning rate for AdamW.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory.")
    parser.add_argument("--bf16", action="store_true", help="Whether to use bf16 (mixed) precision instead of float32 or fp16.")
    
    args = parser.parse_args()

    # Setup logging
    logger.info(f"Arguments: {args}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Load dataset
    # The ALFWorld data should be in a similar format to the LLaDA data:
    # {"trajectory": [{"observation": "...", "action": "..."}, ...]}
    # We need to tokenize this for causal language modeling.
    # For now, we'll assume a 'text' field that contains the full trajectory as a string.
    # This will need to be properly formatted from the JSONL.
    raw_datasets = load_dataset("json", data_files=args.train_file)
    if "train" not in raw_datasets:
        raise ValueError("Dataset must contain a 'train' split.")
    
    # Preprocess dataset
    def preprocess_function(examples):
        # Flatten the trajectory into a single string for causal LM
        # Example: "Observation: ... Action: ... Observation: ..."
        texts = []
        for trajectory_list in examples["trajectory"]:
            full_text = ""
            for item in trajectory_list:
                if "observation" in item:
                    full_text += f"Observation: {item['observation']} "
                if "action" in item:
                    full_text += f"Action: {item['action']} "
            texts.append(full_text.strip())
        
        return tokenizer(texts, truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    # Data collator
    # For causal LM, we can use DataCollatorForLanguageModeling with mlm=False
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        overwrite_output_dir=args.overwrite_output_dir,
        bf16=args.bf16,
        # Other arguments for distributed training if needed
        # deepspeed=..., local_rank=..., report_to="wandb", etc.
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    logger.info("*** Train ***")
    trainer.train()
    trainer.save_model(args.output_dir)
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    main()
