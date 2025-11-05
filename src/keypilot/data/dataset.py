"""
Dataset class for KeyPilot training data.
"""

import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from loguru import logger
from transformers import BertTokenizer


class KeyPilotDataset(Dataset):
    """
    PyTorch Dataset for KeyPilot training.
    
    Supports the pickle format with the following structure:
    - Task types: error (纠错), completion (补全), suggestion (建议)
    - Screen images (visual context)
    - Tokenized text (BERT tokenizer with ## subwords)
    - Task and layout labels
    """
    
    # Task label mapping
    TASK_LABELS = {
        0: "suggestion",    # 智能建议
        1: "completion",    # 自动补全
        2: "error"          # 错误纠正
    }
    
    # Target type mapping
    TARGET_TYPES = {
        0: "chinese",       # 中文
        1: "english",       # 英文
        2: "number",        # 数字
        3: "punctuation",   # 标点
        4: "emoji"          # 表情
    }
    
    def __init__(
        self,
        data_path: str,
        vocab_path: Optional[str] = None,
        max_seq_length: int = 128,
        image_size: tuple = (512, 256),
        max_samples: Optional[int] = None,
        use_bert_tokenizer: bool = True,
    ):
        """
        Initialize KeyPilot dataset.
        
        Args:
            data_path: Path to pickle file (e.g., all_samples_343.pkl)
            vocab_path: Path to vocabulary/tokenizer. If None, uses BERT tokenizer
            max_seq_length: Maximum sequence length for input tokens
            image_size: Target image size (H, W)
            max_samples: Optional maximum number of samples to load
            use_bert_tokenizer: Whether to use BERT tokenizer (recommended: True)
            
        Raises:
            FileNotFoundError: If data_path doesn't exist
            ValueError: If dataset is empty or invalid
        """
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        self.data_path = data_file
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        
        # Get base directory for images
        self.base_dir = data_file.parent
        
        # Initialize tokenizer
        if use_bert_tokenizer:
            # Use pretrained BERT tokenizer (same as data generation)
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            logger.info("Using BERT multilingual tokenizer")
        elif vocab_path:
            from ..utils.vocabulary import KeyPilotVocabulary
            self.tokenizer = KeyPilotVocabulary()
            self.tokenizer.load(vocab_path)
            logger.info(f"Loaded custom tokenizer from {vocab_path}")
        else:
            raise ValueError("Either use_bert_tokenizer=True or provide vocab_path")
        
        # Load data
        logger.info(f"Loading dataset from {data_path}...")
        try:
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load pickle file: {e}")
            raise ValueError(f"Invalid pickle file: {e}")
        
        if not isinstance(self.data, list):
            raise ValueError("Dataset must be a list of samples")
        
        if not self.data:
            raise ValueError("Dataset is empty")
        
        # Limit samples if specified
        if max_samples is not None and max_samples > 0:
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples")
        
        # Count task distribution
        task_counts = {0: 0, 1: 0, 2: 0}
        for sample in self.data:
            task_label = sample.get('task_label', 1)
            task_counts[task_label] = task_counts.get(task_label, 0) + 1
        
        logger.info(f"Loaded {len(self.data)} samples")
        logger.info(f"Task distribution:")
        logger.info(f"  - Suggestion (0): {task_counts[0]} ({task_counts[0]/len(self.data)*100:.1f}%)")
        logger.info(f"  - Completion (1): {task_counts[1]} ({task_counts[1]/len(self.data)*100:.1f}%)")
        logger.info(f"  - Error (2): {task_counts[2]} ({task_counts[2]/len(self.data)*100:.1f}%)")
        logger.info(f"Using images from {self.base_dir}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: Screen image tensor [3, H, W]
                - input_ids: Token IDs [max_seq_length]
                - attention_mask: Attention mask [max_seq_length]
                - task_label: Task type (0:suggestion, 1:completion, 2:error)
                - layout_label: Keyboard layout label
                - target_token_id: Target token ID (single token for completion/error)
                - target_type: Token type (0:中文, 1:英文, 2:数字, 3:标点, 4:表情)
                - For suggestion task: target_token_ids (multiple tokens)
            
        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range [0, {len(self.data)})")
        
        sample = self.data[idx]
        
        # 1. Load and process image
        image_path = self.base_dir / sample['image_path']
        if image_path.exists():
            try:
                image = Image.open(image_path).convert('RGB')
                # Resize to target size
                image = image.resize((self.image_size[1], self.image_size[0]))  # (W, H)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}. Using black image.")
                image = Image.new('RGB', (self.image_size[1], self.image_size[0]), color='black')
        else:
            logger.warning(f"Image not found: {image_path}. Using black image.")
            image = Image.new('RGB', (self.image_size[1], self.image_size[0]), color='black')
        
        # Convert image to tensor [3, H, W]
        import torchvision.transforms as transforms
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = image_transform(image)
        
        # 2. Process input tokens
        input_tokens = sample.get('input_tokens', [])
        
        # Convert tokens to text (BERT tokenizer can handle this)
        # Join tokens and let BERT tokenizer re-tokenize
        input_text = self._tokens_to_text(input_tokens)
        
        # Tokenize with BERT
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)  # [max_seq_length]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_seq_length]
        
        # 3. Process target
        task_label = sample.get('task_label', 1)
        max_target_length = 32  # Unified max length for all tasks
        
        if task_label == 0:  # Suggestion task
            # Multiple target tokens
            target_tokens = sample.get('target_tokens', [])
            target_text = self._tokens_to_text(target_tokens)
            target_encoded = self.tokenizer(
                target_text,
                max_length=max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target_token_ids = target_encoded['input_ids'].squeeze(0)  # [32]
            target_length = (target_encoded['attention_mask'].squeeze(0) == 1).sum().item()
        else:  # Completion or Error task
            # Single target token - pad to max_target_length for consistent batching
            target_token = sample.get('target_token', '')
            target_token_id = self.tokenizer.convert_tokens_to_ids(target_token)
            if target_token_id is None:
                target_token_id = self.tokenizer.unk_token_id
            
            # Create padded tensor [32] with target at first position
            target_token_ids = torch.full((max_target_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            target_token_ids[0] = target_token_id
            target_length = 1
        
        # 4. Other labels
        layout_label = sample.get('layout_label', 0)
        target_type = sample.get('target_type', 1)
        
        # 5. Build output dictionary
        # Note: All samples must have the same keys for DataLoader batching
        output = {
            'image': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'task_label': torch.tensor(task_label, dtype=torch.long),
            'layout_label': torch.tensor(layout_label, dtype=torch.long),
            'target_token_ids': target_token_ids,  # [32] for all tasks
            'target_length': torch.tensor(target_length, dtype=torch.long),  # Actual number of target tokens
            'target_type': torch.tensor(target_type, dtype=torch.long),
            # Add typo fields for all samples (only meaningful for error tasks)
            'typo_position': torch.tensor(sample.get('typo_position', -1), dtype=torch.long),
        }
        
        return output
    
    def _tokens_to_text(self, tokens: List[str]) -> str:
        """
        Convert BERT-style tokens back to text.
        
        Args:
            tokens: List of tokens (with ## for subwords)
            
        Returns:
            Reconstructed text string
        """
        if not tokens:
            return ""
        
        # Join tokens, handling ## subword markers
        text = ""
        for token in tokens:
            if token.startswith('##'):
                # Subword: append without space
                text += token[2:]
            else:
                # New word: add space if not first token
                if text:
                    text += " "
                text += token
        
        return text.strip()
    
    def get_task_name(self, task_label: int) -> str:
        """Get task name from label."""
        return self.TASK_LABELS.get(task_label, "unknown")
    
    def get_target_type_name(self, target_type: int) -> str:
        """Get target type name from label."""
        return self.TARGET_TYPES.get(target_type, "unknown")

