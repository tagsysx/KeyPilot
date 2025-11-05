"""
Vocabulary utilities for KeyPilot.

Implements multilingual BPE tokenization for 32K vocabulary covering:
- English words
- Chinese characters (Hanzi/Pinyin)
- Symbols and numbers
- Emojis
"""

import os
from typing import List, Optional, Dict
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


class KeyPilotVocabulary:
    """
    Multilingual BPE vocabulary for KeyPilot.
    """
    
    SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "<ERR>",     # Error correction task
        "<COMP>",    # Completion task
        "<SUG>",     # Suggestion task
        "<EN>",      # English layout
        "<ZH>",      # Chinese layout
        "<SYM>",     # Symbol layout
        "<EMOJI>",   # Emoji layout
        "<NUM>",     # Number layout
    ]
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train(self,
              files: List[str],
              output_dir: str,
              vocab_size: Optional[int] = None) -> None:
        """
        Train BPE tokenizer on multilingual corpus.
        
        Args:
            files: List of training file paths
            output_dir: Directory to save tokenizer
            vocab_size: Vocabulary size (default: self.vocab_size)
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # Initialize tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # Setup trainer
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
            initial_alphabet=[],  # Learn from data
            min_frequency=2
        )
        
        # Train
        print(f"Training BPE tokenizer on {len(files)} files...")
        tokenizer.train(files, trainer)
        
        # Add post-processing
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )
        
        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path / "tokenizer.json"))
        
        print(f"Tokenizer saved to {output_path / 'tokenizer.json'}")
        self.tokenizer = tokenizer
    
    def load(self, tokenizer_path: str) -> None:
        """Load pre-trained tokenizer."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    
    def encode(self, text: str, max_length: int = 64) -> Dict[str, List[int]]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load() or train() first.")
        
        encoding = self.tokenizer.encode(text)
        
        # Truncate or pad
        input_ids = encoding.ids[:max_length]
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        pad_id = self.tokenizer.token_to_id("[PAD]")
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids += [pad_id] * padding_length
            attention_mask += [0] * padding_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.tokenizer is None:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        
        return {
            token: self.tokenizer.token_to_id(token)
            for token in self.SPECIAL_TOKENS
        }


def create_multilingual_corpus(
    english_files: List[str],
    chinese_files: List[str],
    output_file: str,
    ratio: float = 0.5) -> None:
    """
    Create balanced multilingual corpus for tokenizer training.
    
    Args:
        english_files: English text files
        chinese_files: Chinese text files
        output_file: Output corpus file
        ratio: English to Chinese ratio
    """
    import random
    
    print("Creating multilingual corpus...")
    
    # Read English data
    english_lines = []
    for file in english_files:
        with open(file, 'r', encoding='utf-8') as f:
            english_lines.extend(f.readlines())
    
    # Read Chinese data
    chinese_lines = []
    for file in chinese_files:
        with open(file, 'r', encoding='utf-8') as f:
            chinese_lines.extend(f.readlines())
    
    # Balance
    en_count = int(len(english_lines) * ratio)
    zh_count = int(len(chinese_lines) * (1 - ratio))
    
    english_sample = random.sample(english_lines, min(en_count, len(english_lines)))
    chinese_sample = random.sample(chinese_lines, min(zh_count, len(chinese_lines)))
    
    # Shuffle and write
    combined = english_sample + chinese_sample
    random.shuffle(combined)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(combined)
    
    print(f"Created corpus with {len(combined)} lines at {output_file}")


if __name__ == "__main__":
    # Example usage
    vocab = KeyPilotVocabulary(vocab_size=32000)
    
    # Train on corpus (placeholder)
    # corpus_files = ["data/multilingual_corpus.txt"]
    # vocab.train(corpus_files, "models/tokenizer")
    
    # Or load existing
    # vocab.load("models/tokenizer/tokenizer.json")
    
    # Test encoding
    # result = vocab.encode("Hello 世界! 😊")
    # print(result)

