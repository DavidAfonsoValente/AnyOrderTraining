import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import get_scheduler, PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import wandb
from typing import Optional, Dict, Any
from omegaconf import DictConfig

from .losses import masked_cross_entropy_loss

class AOMTTrainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: torch.utils.data.Dataset,
        config: DictConfig,
        method: str,
    ):
        self.config = config
        self.method = method
        self.tokenizer = tokenizer
        
        # Initialize Accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            log_with="wandb" if config.get("use_wandb", False) else None,
        )
        
        if self.accelerator.is_main_process and config.get("use_wandb", False):
            self.accelerator.init_trackers(
                project_name=config.get("wandb_project", "aomt"),
                config=dict(config),
                init_kwargs={"wandb": {"name": config.get("run_name", method)}}
            )

        self.model = model
        self.train_dataset = train_dataset
        
        # Optimizer & Scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # Prepare for distributed training
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=config.collator_fn # Passed from outside to avoid circular deps
        )
        
        num_update_steps_per_epoch = len(self.train_dataloader) // config.get("gradient_accumulation_steps", 1)
        max_train_steps = config.epochs * num_update_steps_per_epoch
        
        self.lr_scheduler = get_scheduler(
            name=config.lr_schedule,
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=max_train_steps,
        )
        
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def train(self):
        self.model.train()
        total_steps = 0
        
        for epoch in range(self.config.epochs):
            # Critical for AOMT: resample masks every epoch
            if hasattr(self.train_dataset, "set_epoch"):
                self.train_dataset.set_epoch(epoch)
                
            progress_bar = tqdm(
                range(len(self.train_dataloader)), 
                disable=not self.accelerator.is_local_main_process,
                desc=f"Epoch {epoch}"
            )
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                    logits = outputs.logits
                    loss = masked_cross_entropy_loss(logits, batch["labels"])
                    
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                if self.accelerator.is_main_process:
                    metrics = {
                        "loss": loss.item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": total_steps
                    }
                    self.accelerator.log(metrics, step=total_steps)
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item()})
                
                total_steps += 1
                
                if total_steps % self.config.checkpoint_save_steps == 0:
                    self.save_checkpoint(total_steps)
            
            # Save end of epoch checkpoint
            self.save_checkpoint(total_steps, is_epoch_end=True)

    def save_checkpoint(self, step: int, is_epoch_end: bool = False):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            suffix = f"step_{step}" if not is_epoch_end else f"epoch_end"
            output_dir = os.path.join(self.config.output_dir, f"checkpoint-{suffix}")
            # Use accelerator.save_state or manual model save
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                output_dir, 
                is_main_process=self.accelerator.is_main_process, 
                save_function=self.accelerator.save
            )
            self.tokenizer.save_pretrained(output_dir)
            print(f"Saved checkpoint to {output_dir}")
