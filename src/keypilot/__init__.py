"""
KeyPilot: A Vision-Language Typing Agent for Intelligent Keyboard Prediction

This package provides tools for training and deploying vision-language models
that predict keyboard input intent and optimal layouts based on screen context
and conversation history.
"""

__version__ = "0.1.0"
__author__ = "KeyPilot Team"

from .models import KeyPilotVLM, KeyPilotEncoder, KeyPilotDecoder, KeyPilotLoss
from .data.dataset import KeyPilotDataset
from .evaluation.evaluator import KeyPilotEvaluator

__all__ = [
    "KeyPilotVLM",
    "KeyPilotEncoder",
    "KeyPilotDecoder",
    "KeyPilotLoss",
    "KeyPilotDataset",
    "KeyPilotEvaluator",
]

