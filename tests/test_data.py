"""
Tests for data module.
"""

import pytest
import json
import tempfile
from pathlib import Path
from keypilot.data.preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Tests for DataPreprocessor."""
    
    def test_default_splits(self):
        """Test default split configuration."""
        preprocessor = DataPreprocessor()
        assert preprocessor.train_split == 0.8
        assert preprocessor.val_split == 0.1
        assert preprocessor.test_split == 0.1
    
    def test_invalid_splits(self):
        """Test that invalid splits raise error."""
        with pytest.raises(ValueError):
            DataPreprocessor(train_split=0.5, val_split=0.3, test_split=0.1)
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        # Create sample dataset
        sample_data = [
            {
                "conversation_text": f"Sample {i}",
                "screen_description": f"Screen {i}",
                "next_intent": "text",
                "optimal_layout": "qwerty",
            }
            for i in range(100)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample data
            data_path = Path(tmpdir) / "data.json"
            with open(data_path, 'w') as f:
                json.dump(sample_data, f)
            
            # Split dataset
            output_dir = Path(tmpdir) / "split"
            preprocessor = DataPreprocessor()
            train_path, val_path, test_path = preprocessor.split_dataset(
                str(data_path),
                str(output_dir),
            )
            
            # Verify files exist
            assert Path(train_path).exists()
            assert Path(val_path).exists()
            assert Path(test_path).exists()
            
            # Load and verify sizes
            with open(train_path, 'r') as f:
                train_data = json.load(f)
            with open(val_path, 'r') as f:
                val_data = json.load(f)
            with open(test_path, 'r') as f:
                test_data = json.load(f)
            
            assert len(train_data) == 80
            assert len(val_data) == 10
            assert len(test_data) == 10
            assert len(train_data) + len(val_data) + len(test_data) == 100
    
    def test_split_nonexistent_file(self):
        """Test splitting non-existent file raises error."""
        preprocessor = DataPreprocessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                preprocessor.split_dataset(
                    "nonexistent.json",
                    tmpdir,
                )
    
    def test_split_invalid_json(self):
        """Test splitting invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid JSON file
            data_path = Path(tmpdir) / "invalid.json"
            with open(data_path, 'w') as f:
                f.write("not valid json")
            
            preprocessor = DataPreprocessor()
            with pytest.raises(ValueError):
                preprocessor.split_dataset(
                    str(data_path),
                    tmpdir,
                )

