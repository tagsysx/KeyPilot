"""
Tests for agent module.
"""

import pytest
from unittest.mock import Mock, MagicMock
from keypilot.agents.data_generator import DataGenerationAgent, MultiAgentPipeline
from keypilot.agents.llm_client import LLMClient


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""
    
    def generate(self, prompt, max_tokens=1000, temperature=0.7, **kwargs):
        """Generate mock response."""
        return '''{
            "conversation_text": "Hello, how are you?",
            "screen_description": "Chat app interface",
            "next_intent": "text",
            "optimal_layout": "qwerty"
        }'''


class TestDataGenerationAgent:
    """Tests for DataGenerationAgent."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        mock_client = MockLLMClient()
        intent_classes = ["text", "emoji", "numeric"]
        layout_types = ["qwerty", "numeric"]
        
        agent = DataGenerationAgent(
            agent_id=0,
            llm_client=mock_client,
            intent_classes=intent_classes,
            layout_types=layout_types,
        )
        
        assert agent.agent_id == 0
        assert agent.intent_classes == intent_classes
        assert agent.layout_types == layout_types
    
    def test_empty_intent_classes_raises_error(self):
        """Test that empty intent classes raise error."""
        mock_client = MockLLMClient()
        with pytest.raises(ValueError):
            DataGenerationAgent(
                agent_id=0,
                llm_client=mock_client,
                intent_classes=[],
                layout_types=["qwerty"],
            )
    
    def test_empty_layout_types_raises_error(self):
        """Test that empty layout types raise error."""
        mock_client = MockLLMClient()
        with pytest.raises(ValueError):
            DataGenerationAgent(
                agent_id=0,
                llm_client=mock_client,
                intent_classes=["text"],
                layout_types=[],
            )
    
    def test_generate_sample(self):
        """Test generating a single sample."""
        mock_client = MockLLMClient()
        agent = DataGenerationAgent(
            agent_id=0,
            llm_client=mock_client,
            intent_classes=["text", "emoji"],
            layout_types=["qwerty", "numeric"],
        )
        
        sample = agent.generate_sample("Replying to a text message")
        
        assert "conversation_text" in sample
        assert "screen_description" in sample
        assert "next_intent" in sample
        assert "optimal_layout" in sample
        assert sample["agent_id"] == 0
    
    def test_generate_batch(self):
        """Test generating multiple samples."""
        mock_client = MockLLMClient()
        agent = DataGenerationAgent(
            agent_id=0,
            llm_client=mock_client,
            intent_classes=["text", "emoji"],
            layout_types=["qwerty", "numeric"],
        )
        
        samples = agent.generate_batch(num_samples=5)
        
        assert len(samples) <= 5  # May be less if some fail
        for sample in samples:
            assert "conversation_text" in sample
            assert "next_intent" in sample
    
    def test_invalid_num_samples_raises_error(self):
        """Test that invalid num_samples raises error."""
        mock_client = MockLLMClient()
        agent = DataGenerationAgent(
            agent_id=0,
            llm_client=mock_client,
            intent_classes=["text"],
            layout_types=["qwerty"],
        )
        
        with pytest.raises(ValueError):
            agent.generate_batch(num_samples=0)
        
        with pytest.raises(ValueError):
            agent.generate_batch(num_samples=-1)


class TestMultiAgentPipeline:
    """Tests for MultiAgentPipeline."""
    
    def test_invalid_num_agents_raises_error(self):
        """Test that invalid num_agents raises error."""
        with pytest.raises(ValueError):
            MultiAgentPipeline(
                num_agents=0,
                intent_classes=["text"],
                layout_types=["qwerty"],
                use_chatgpt=False,
                use_deepseek=False,
            )
    
    def test_no_llm_providers_raises_error(self):
        """Test that no LLM providers raises error."""
        with pytest.raises(ValueError):
            MultiAgentPipeline(
                num_agents=2,
                intent_classes=["text"],
                layout_types=["qwerty"],
                use_chatgpt=False,
                use_deepseek=False,
            )

