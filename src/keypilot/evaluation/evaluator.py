"""
Evaluation utilities for KeyPilot model.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from ..models.model import KeyPilotVLM
from ..data.dataset import KeyPilotDataset


class KeyPilotEvaluator:
    """Evaluator for KeyPilot model performance."""
    
    def __init__(
        self,
        model: KeyPilotVLM,
        test_dataset: KeyPilotDataset,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: KeyPilot model to evaluate
            test_dataset: Test dataset
            device: Device to run evaluation on
            
        Raises:
            ValueError: If test_dataset is empty
        """
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty")
        
        self.model = model
        self.test_dataset = test_dataset
        self.device = device
        
        self.model.to(device)
        self.model.eval()
        
        logger.info(f"Evaluator initialized with {len(test_dataset)} test samples")
    
    @torch.no_grad()
    def evaluate(self, batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
            
        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        logger.info("Starting evaluation")
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # Collect predictions and labels
        intent_preds = []
        intent_labels = []
        layout_preds = []
        layout_labels = []
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                pixel_values=batch.get('pixel_values'),
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
            )
            
            # Get predictions
            intent_pred = torch.argmax(outputs['intent_logits'], dim=-1)
            layout_pred = torch.argmax(outputs['layout_logits'], dim=-1)
            
            # Collect results
            intent_preds.extend(intent_pred.cpu().numpy())
            intent_labels.extend(batch['intent_label'].cpu().numpy())
            layout_preds.extend(layout_pred.cpu().numpy())
            layout_labels.extend(batch['layout_label'].cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            intent_preds, intent_labels,
            layout_preds, layout_labels,
        )
        
        logger.info("Evaluation completed")
        self._log_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(
        self,
        intent_preds: List[int],
        intent_labels: List[int],
        layout_preds: List[int],
        layout_labels: List[int],
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        # Intent metrics
        intent_accuracy = accuracy_score(intent_labels, intent_preds)
        intent_precision, intent_recall, intent_f1, _ = precision_recall_fscore_support(
            intent_labels, intent_preds, average='weighted', zero_division=0
        )
        
        # Layout metrics
        layout_accuracy = accuracy_score(layout_labels, layout_preds)
        layout_precision, layout_recall, layout_f1, _ = precision_recall_fscore_support(
            layout_labels, layout_preds, average='weighted', zero_division=0
        )
        
        # Confusion matrices
        intent_cm = confusion_matrix(intent_labels, intent_preds)
        layout_cm = confusion_matrix(layout_labels, layout_preds)
        
        # Classification reports
        intent_classes = self.test_dataset.intent_classes
        layout_types = self.test_dataset.layout_types
        
        intent_report = classification_report(
            intent_labels, intent_preds,
            target_names=intent_classes,
            output_dict=True,
            zero_division=0,
        )
        
        layout_report = classification_report(
            layout_labels, layout_preds,
            target_names=layout_types,
            output_dict=True,
            zero_division=0,
        )
        
        return {
            'intent': {
                'accuracy': intent_accuracy,
                'precision': intent_precision,
                'recall': intent_recall,
                'f1': intent_f1,
                'confusion_matrix': intent_cm.tolist(),
                'classification_report': intent_report,
            },
            'layout': {
                'accuracy': layout_accuracy,
                'precision': layout_precision,
                'recall': layout_recall,
                'f1': layout_f1,
                'confusion_matrix': layout_cm.tolist(),
                'classification_report': layout_report,
            },
            'overall': {
                'mean_accuracy': (intent_accuracy + layout_accuracy) / 2,
                'mean_f1': (intent_f1 + layout_f1) / 2,
            }
        }
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log evaluation metrics."""
        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        
        logger.info("\nIntent Classification:")
        logger.info(f"  Accuracy:  {metrics['intent']['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['intent']['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['intent']['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['intent']['f1']:.4f}")
        
        logger.info("\nLayout Classification:")
        logger.info(f"  Accuracy:  {metrics['layout']['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['layout']['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['layout']['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['layout']['f1']:.4f}")
        
        logger.info("\nOverall Performance:")
        logger.info(f"  Mean Accuracy: {metrics['overall']['mean_accuracy']:.4f}")
        logger.info(f"  Mean F1:       {metrics['overall']['mean_f1']:.4f}")
        logger.info("=" * 50)
    
    def save_results(self, metrics: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            metrics: Evaluation metrics dictionary
            output_path: Path to save results
            
        Raises:
            IOError: If results cannot be saved
        """
        import json
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except IOError as e:
            logger.error(f"Failed to save results: {e}")
            raise IOError(f"Failed to save results: {e}")

