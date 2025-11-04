"""
Training pipeline for KeyPilot model.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_cosine_schedule_with_warmup
from transformers import get_scheduler
from typing import Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None

from ..models.vlm import KeyPilotVLM
from ..data.dataset import KeyPilotDataset
from ..utils.config import TrainingConfig


class KeyPilotTrainer:
    """Trainer for KeyPilot vision-language model."""
    
    def __init__(
        self,
        model: KeyPilotVLM,
        train_dataset: KeyPilotDataset,
        val_dataset: KeyPilotDataset,
        config: TrainingConfig,
    ):
        """
        Initialize trainer.
        
        Args:
            model: KeyPilot model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            
        Raises:
            ValueError: If datasets are empty
        """
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty")
        if len(val_dataset) == 0:
            raise ValueError("Validation dataset is empty")
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        total_steps = len(self.train_loader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Setup wandb if enabled
        self.use_wandb = config.use_wandb and wandb is not None
        if self.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__,
            )
            logger.info("Weights & Biases logging enabled")
        elif config.use_wandb and wandb is None:
            logger.warning("wandb requested but not installed, skipping")
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        logger.info("Trainer initialized")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Total training steps: {total_steps}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        
        return optimizer
    
    def train(self) -> None:
        """
        Train the model.
        
        Raises:
            RuntimeError: If training fails
        """
        logger.info("Starting training")
        
        try:
            for epoch in range(self.config.num_epochs):
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Training
                train_metrics = self._train_epoch(epoch)
                logger.info(f"Train loss: {train_metrics['loss']:.4f}")
                
                # Validation
                val_metrics = self._validate()
                logger.info(f"Validation loss: {val_metrics['loss']:.4f}")
                logger.info(f"Validation intent accuracy: {val_metrics['intent_accuracy']:.4f}")
                logger.info(f"Validation layout accuracy: {val_metrics['layout_accuracy']:.4f}")
                
                # Save checkpoint
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self._save_checkpoint('best_model')
                    logger.info("Saved best model checkpoint")
                
                if (epoch + 1) % 5 == 0:
                    self._save_checkpoint(f'checkpoint_epoch_{epoch + 1}')
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")
        finally:
            if self.use_wandb:
                wandb.finish()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch.get('pixel_values'),
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels={
                    'intent': batch['intent_label'],
                    'layout': batch['layout_label'],
                }
            )
            
            loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    if self.use_wandb:
                        wandb.log({
                            'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                            'train/learning_rate': self.scheduler.get_last_lr()[0],
                            'train/step': self.global_step,
                        })
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            progress_bar.set_postfix({'loss': total_loss / num_batches})
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        intent_correct = 0
        layout_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch.get('pixel_values'),
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels={
                    'intent': batch['intent_label'],
                    'layout': batch['layout_label'],
                }
            )
            
            total_loss += outputs['loss'].item()
            
            # Calculate accuracy
            intent_preds = torch.argmax(outputs['intent_logits'], dim=-1)
            layout_preds = torch.argmax(outputs['layout_logits'], dim=-1)
            
            intent_correct += (intent_preds == batch['intent_label']).sum().item()
            layout_correct += (layout_preds == batch['layout_label']).sum().item()
            total_samples += batch['intent_label'].size(0)
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'intent_accuracy': intent_correct / total_samples,
            'layout_accuracy': layout_correct / total_samples,
        }
        
        if self.use_wandb:
            wandb.log({
                'val/loss': metrics['loss'],
                'val/intent_accuracy': metrics['intent_accuracy'],
                'val/layout_accuracy': metrics['layout_accuracy'],
            })
        
        return metrics
    
    def _save_checkpoint(self, name: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name
            
        Raises:
            IOError: If checkpoint cannot be saved
        """
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model.save_pretrained(str(output_dir))
            logger.info(f"Checkpoint saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise IOError(f"Failed to save checkpoint: {e}")

