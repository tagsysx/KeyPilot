"""
Tests for configuration module.
"""

import pytest
import tempfile
from pathlib import Path
from keypilot.utils.config import (
    KeyPilotConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    DistillationConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""
    
    def test_default_config(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.base_model == "microsoft/phi-3-vision-128k-instruct"
        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.load_in_4bit is True
    
    def test_custom_config(self):
        """Test custom model configuration."""
        config = ModelConfig(
            base_model="custom-model",
            use_lora=False,
            load_in_4bit=False,
        )
        assert config.base_model == "custom-model"
        assert config.use_lora is False
        assert config.load_in_4bit is False


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_default_splits(self):
        """Test default data splits."""
        config = DataConfig()
        assert config.train_split == 0.8
        assert config.val_split == 0.1
        assert config.test_split == 0.1
        assert config.train_split + config.val_split + config.test_split == 1.0
    
    def test_num_agents(self):
        """Test number of agents configuration."""
        config = DataConfig(num_agents=8)
        assert config.num_agents == 8


class TestKeyPilotConfig:
    """Tests for main KeyPilotConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = KeyPilotConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.distillation, DistillationConfig)
        assert len(config.intent_classes) > 0
        assert len(config.layout_types) > 0
    
    def test_intent_classes(self):
        """Test intent classes."""
        config = KeyPilotConfig()
        expected_intents = ["text", "symbol", "emoji", "numeric"]
        for intent in expected_intents:
            assert intent in config.intent_classes
    
    def test_layout_types(self):
        """Test layout types."""
        config = KeyPilotConfig()
        expected_layouts = ["qwerty", "numeric", "symbol", "emoji"]
        for layout in expected_layouts:
            assert layout in config.layout_types
    
    def test_save_and_load_yaml(self):
        """Test saving and loading configuration from YAML."""
        config = KeyPilotConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            
            # Save configuration
            config.to_yaml(str(config_path))
            assert config_path.exists()
            
            # Load configuration
            loaded_config = KeyPilotConfig.from_yaml(str(config_path))
            assert loaded_config.model.base_model == config.model.base_model
            assert loaded_config.training.num_epochs == config.training.num_epochs
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            KeyPilotConfig.from_yaml("nonexistent.yaml")

