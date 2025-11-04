"""
Tests for models module.
"""

import pytest
from keypilot.utils.config import ModelConfig


class TestModelConfig:
    """Tests for model configuration and initialization."""
    
    def test_model_config_defaults(self):
        """Test default model configuration values."""
        config = ModelConfig()
        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.load_in_4bit is True
    
    def test_model_config_custom(self):
        """Test custom model configuration."""
        config = ModelConfig(
            base_model="custom-model",
            lora_r=32,
            load_in_4bit=False,
        )
        assert config.base_model == "custom-model"
        assert config.lora_r == 32
        assert config.load_in_4bit is False


# Note: Full model tests require GPU and would be slow
# These would be integration tests run separately
class TestKeyPilotVLMIntegration:
    """
    Integration tests for KeyPilotVLM.
    
    These tests require GPU and are skipped in CI.
    Run with: pytest -m integration
    """
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_model_initialization(self):
        """Test model can be initialized (requires GPU)."""
        from keypilot.models.vlm import KeyPilotVLM
        from keypilot.utils.config import ModelConfig
        
        config = ModelConfig()
        intent_classes = ["text", "emoji", "numeric"]
        layout_types = ["qwerty", "numeric"]
        
        model = KeyPilotVLM(config, intent_classes, layout_types)
        assert model is not None
    
    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires GPU and model download")
    def test_model_forward_pass(self):
        """Test model forward pass (requires GPU)."""
        import torch
        from keypilot.models.vlm import KeyPilotVLM
        from keypilot.utils.config import ModelConfig
        
        config = ModelConfig()
        intent_classes = ["text", "emoji", "numeric"]
        layout_types = ["qwerty", "numeric"]
        
        model = KeyPilotVLM(config, intent_classes, layout_types)
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 50))
        
        outputs = model(pixel_values, input_ids)
        
        assert 'intent_logits' in outputs
        assert 'layout_logits' in outputs
        assert outputs['intent_logits'].shape[0] == batch_size
        assert outputs['layout_logits'].shape[0] == batch_size

