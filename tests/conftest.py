"""
Pytest configuration and fixtures.
"""

import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def sample_data():
    """Fixture providing sample training data."""
    return [
        {
            "conversation_text": "Hey, how are you?",
            "screen_description": "Messaging app",
            "next_intent": "text",
            "optimal_layout": "qwerty",
        },
        {
            "conversation_text": "Call me at ",
            "screen_description": "Contact form",
            "next_intent": "numeric",
            "optimal_layout": "numeric",
        },
        {
            "conversation_text": "I love this! ",
            "screen_description": "Social media post",
            "next_intent": "emoji",
            "optimal_layout": "emoji",
        },
    ]


@pytest.fixture
def temp_data_file(sample_data):
    """Fixture providing temporary data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    """Fixture providing temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def intent_classes():
    """Fixture providing intent classes."""
    return ["text", "symbol", "emoji", "numeric", "space", "enter", "delete"]


@pytest.fixture
def layout_types():
    """Fixture providing layout types."""
    return ["qwerty", "numeric", "symbol", "emoji"]

