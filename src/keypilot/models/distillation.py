"""
Knowledge distillation for creating lightweight on-device models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from loguru import logger

from ..utils.config import DistillationConfig


class DistillationTrainer:
    """
    Knowledge distillation trainer.
    
    Distills a large teacher model into a smaller student model
    for efficient on-device deployment.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: DistillationConfig,
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_model: Large pre-trained teacher model
            student_model: Smaller student model to train
            config: Distillation configuration
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info("DistillationTrainer initialized")
        logger.info(f"Temperature: {config.distillation_temperature}")
        logger.info(f"Alpha: {config.distillation_alpha}")
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        alpha: float,
    ) -> torch.Tensor:
        """
        Calculate distillation loss.
        
        Combines soft targets from teacher with hard labels.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss vs hard label loss
            
        Returns:
            Combined loss
        """
        # Soft target loss (KL divergence between teacher and student)
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
        soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Hard label loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return loss
    
    @torch.no_grad()
    def evaluate_student(
        self,
        test_loader: DataLoader,
        device: str = "cuda",
    ) -> float:
        """
        Evaluate student model accuracy.
        
        Args:
            test_loader: Test data loader
            device: Device to run on
            
        Returns:
            Accuracy score
        """
        self.student_model.eval()
        correct = 0
        total = 0
        
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = self.student_model(
                pixel_values=batch.get('pixel_values'),
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
            )
            
            # Check intent predictions
            intent_preds = torch.argmax(outputs['intent_logits'], dim=-1)
            correct += (intent_preds == batch['intent_label']).sum().item()
            total += batch['intent_label'].size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def save_student(self, output_dir: str) -> None:
        """
        Save distilled student model.
        
        Args:
            output_dir: Directory to save model
            
        Raises:
            IOError: If model cannot be saved
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.student_model.save_pretrained(str(output_path))
            logger.info(f"Student model saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save student model: {e}")
            raise IOError(f"Failed to save student model: {e}")

