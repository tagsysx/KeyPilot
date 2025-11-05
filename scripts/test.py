"""
Test script for KeyPilot model.

This script tests a trained KeyPilot model on test dataset.

Usage:
    python scripts/test.py --checkpoint results/train/models/best_model --data data/raw/data/test/all_samples_343.pkl --output_test results/test
"""

import sys
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keypilot.models import KeyPilotVLM
from keypilot.data.dataset import KeyPilotDataset
from keypilot.evaluation.evaluator import KeyPilotEvaluator
from keypilot.utils.logger import setup_logger
from keypilot.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test KeyPilot model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to test dataset JSON file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing screen images (optional)"
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default="results/test",
        help="Output directory for test/evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (cuda/cpu)"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: str):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        config: Model configuration
        device: Device to load model on
        
    Returns:
        Loaded model
        
    Raises:
        FileNotFoundError: If checkpoint not found
    """
    logger = setup_logger("model_loading")
    
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load model
    try:
        model = KeyPilotModel.from_pretrained(checkpoint_path)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def create_test_dataset(
    data_path: str,
    processor,
    config: dict,
    image_dir: str = None
):
    """
    Create test dataset.
    
    Args:
        data_path: Path to test JSON file
        processor: Data processor
        config: Configuration dict
        image_dir: Optional image directory
        
    Returns:
        Test dataset
    """
    logger = setup_logger("dataset_creation")
    
    logger.info(f"Loading test dataset from {data_path}")
    
    # Get intent and layout classes from config
    intent_classes = config.get('intent_classes', ['text', 'number', 'email', 'url', 'search'])
    layout_types = config.get('layout_types', ['qwerty', 'numeric', 'symbol', 'emoji', 'url'])
    
    # Create dataset
    test_dataset = KeyPilotDataset(
        data_path=data_path,
        processor=processor,
        intent_classes=intent_classes,
        layout_types=layout_types,
        image_dir=image_dir,
    )
    
    logger.info(f"Test dataset created with {len(test_dataset)} samples")
    return test_dataset


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("evaluation", Path(args.output).parent)
    
    logger.info("=" * 60)
    logger.info("KeyPilot Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Test data: {args.data}")
    logger.info(f"Device: {args.device}")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Create processor
    logger.info("Initializing processor...")
    try:
        # Use a simple processor for now
        # In production, use the appropriate processor for your model
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained("microsoft/git-base")
        logger.info("Processor initialized")
    except Exception as e:
        logger.warning(f"Failed to load standard processor: {e}")
        logger.info("Creating dummy processor")
        processor = DummyProcessor()
    
    # Load model
    try:
        model = load_model(args.checkpoint, config, args.device)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return
    
    # Create test dataset
    try:
        test_dataset = create_test_dataset(
            args.data,
            processor,
            config,
            args.image_dir
        )
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        return
    
    # Create evaluator
    try:
        evaluator = KeyPilotEvaluator(
            model=model,
            test_dataset=test_dataset,
            device=args.device
        )
        logger.info("Evaluator created")
    except Exception as e:
        logger.error(f"Failed to create evaluator: {e}")
        return
    
    # Create output directory
    output_dir = Path(args.output_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Test output directory: {output_dir}")
    
    # Run evaluation
    try:
        logger.info("Starting evaluation...")
        metrics = evaluator.evaluate(batch_size=args.batch_size)
        
        # Save results
        output_file = output_dir / "evaluation_results.json"
        evaluator.save_results(metrics, str(output_file))
        
        logger.info("=" * 60)
        logger.info("Evaluation completed successfully!")
        logger.info(f"Results saved to {output_file}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


class DummyProcessor:
    """Dummy processor for testing when standard processor is not available."""
    
    def __call__(self, text, images, return_tensors="pt", padding="max_length", truncation=True):
        """Process inputs."""
        import torch
        # Create dummy tensors
        batch_size = 1
        return {
            'input_ids': torch.zeros((batch_size, 64), dtype=torch.long),
            'attention_mask': torch.ones((batch_size, 64), dtype=torch.long),
            'pixel_values': torch.randn((batch_size, 3, 224, 224)),
        }


if __name__ == "__main__":
    main()

