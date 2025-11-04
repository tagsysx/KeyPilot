"""
Data preprocessing utilities.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from loguru import logger


class DataPreprocessor:
    """Preprocess and split KeyPilot dataset."""
    
    def __init__(
        self,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize data preprocessor.
        
        Args:
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If splits don't sum to 1.0
        """
        total = train_split + val_split + test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {total}")
        
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        logger.info(f"DataPreprocessor initialized with splits: train={train_split}, val={val_split}, test={test_split}")
    
    def split_dataset(
        self,
        data_path: str,
        output_dir: str,
    ) -> Tuple[str, str, str]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            data_path: Path to input dataset JSON file
            output_dir: Directory to save split datasets
            
        Returns:
            Tuple of (train_path, val_path, test_path)
            
        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If dataset is invalid
            IOError: If output files cannot be written
        """
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dataset JSON: {e}")
            raise ValueError(f"Invalid JSON in dataset file: {e}")
        
        if not isinstance(data, list) or not data:
            raise ValueError("Dataset must be a non-empty JSON array")
        
        logger.info(f"Loaded {len(data)} samples")
        
        # Split data
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data,
            train_size=self.train_split,
            random_state=self.seed,
            shuffle=True,
        )
        
        # Second split: val vs test
        val_size = self.val_split / (self.val_split + self.test_split)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_size,
            random_state=self.seed,
            shuffle=True,
        )
        
        logger.info(f"Split into train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = output_path / "train.json"
        val_path = output_path / "val.json"
        test_path = output_path / "test.json"
        
        try:
            with open(train_path, 'w') as f:
                json.dump(train_data, f, indent=2)
            logger.info(f"Train set saved to {train_path}")
            
            with open(val_path, 'w') as f:
                json.dump(val_data, f, indent=2)
            logger.info(f"Validation set saved to {val_path}")
            
            with open(test_path, 'w') as f:
                json.dump(test_data, f, indent=2)
            logger.info(f"Test set saved to {test_path}")
            
        except IOError as e:
            logger.error(f"Failed to save split datasets: {e}")
            raise IOError(f"Failed to save split datasets: {e}")
        
        return str(train_path), str(val_path), str(test_path)

