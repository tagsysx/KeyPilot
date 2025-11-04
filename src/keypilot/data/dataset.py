"""
Dataset class for KeyPilot training data.
"""

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Any, Optional
from pathlib import Path
from loguru import logger


class KeyPilotDataset(Dataset):
    """
    PyTorch Dataset for KeyPilot training.
    
    Each sample contains:
    - Screen image (visual context)
    - Conversation text (linguistic context)
    - Next input intent label
    - Optimal keyboard layout label
    """
    
    def __init__(
        self,
        data_path: str,
        processor: Any,
        intent_classes: List[str],
        layout_types: List[str],
        max_samples: Optional[int] = None,
    ):
        """
        Initialize KeyPilot dataset.
        
        Args:
            data_path: Path to JSON dataset file
            processor: Processor for encoding images and text
            intent_classes: List of input intent classes
            layout_types: List of keyboard layout types
            max_samples: Optional maximum number of samples to load
            
        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If dataset is empty or invalid
        """
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        if not intent_classes:
            raise ValueError("intent_classes cannot be empty")
        if not layout_types:
            raise ValueError("layout_types cannot be empty")
        
        self.processor = processor
        self.intent_classes = intent_classes
        self.layout_types = layout_types
        
        # Create label mappings
        self.intent_to_idx = {intent: idx for idx, intent in enumerate(intent_classes)}
        self.layout_to_idx = {layout: idx for idx, layout in enumerate(layout_types)}
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        try:
            with open(data_file, 'r') as f:
                self.data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse dataset JSON: {e}")
            raise ValueError(f"Invalid JSON in dataset file: {e}")
        
        if not isinstance(self.data, list):
            raise ValueError("Dataset must be a JSON array")
        
        if not self.data:
            raise ValueError("Dataset is empty")
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
        
        logger.info(f"Loaded {len(self.data)} samples")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing processed inputs and labels
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")
        
        sample = self.data[idx]
        
        # Note: In practice, you would load actual screen images
        # For now, we create a placeholder image from screen description
        # In real implementation, screen_image_path should be in the data
        image = self._create_placeholder_image(sample.get('screen_description', ''))
        
        # Get text
        conversation_text = sample.get('conversation_text', '')
        
        # Process inputs
        inputs = self.processor(
            text=conversation_text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        
        # Remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Get labels
        intent = sample.get('next_intent', 'text')
        layout = sample.get('optimal_layout', 'qwerty')
        
        # Map to indices (with fallback to default)
        intent_idx = self.intent_to_idx.get(intent, 0)
        layout_idx = self.layout_to_idx.get(layout, 0)
        
        inputs['intent_label'] = torch.tensor(intent_idx, dtype=torch.long)
        inputs['layout_label'] = torch.tensor(layout_idx, dtype=torch.long)
        
        return inputs
    
    def _create_placeholder_image(self, description: str) -> Image.Image:
        """
        Create a placeholder image.
        
        In real implementation, this would load actual screen images.
        
        Args:
            description: Screen description text
            
        Returns:
            PIL Image
        """
        # Create a simple placeholder image
        # In practice, you would load actual screen captures
        return Image.new('RGB', (224, 224), color='white')

