"""
Vision-Language Model for KeyPilot keyboard prediction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
from loguru import logger

from ..utils.config import ModelConfig


class KeyPilotVLM(nn.Module):
    """
    Vision-Language Model for predicting keyboard input intent and layout.
    
    This model takes as input:
    - Visual context: Screenshot of the user's screen
    - Linguistic context: Conversation history
    
    And predicts:
    - Next input intent: text, symbol, emoji, numeric, etc.
    - Optimal keyboard layout: qwerty, numeric, symbol, emoji
    """
    
    def __init__(
        self,
        config: ModelConfig,
        intent_classes: List[str],
        layout_types: List[str],
    ):
        """
        Initialize KeyPilot Vision-Language Model.
        
        Args:
            config: Model configuration
            intent_classes: List of input intent classes
            layout_types: List of keyboard layout types
            
        Raises:
            ValueError: If intent_classes or layout_types are empty
        """
        super().__init__()
        
        if not intent_classes:
            raise ValueError("intent_classes cannot be empty")
        if not layout_types:
            raise ValueError("layout_types cannot be empty")
        
        self.config = config
        self.intent_classes = intent_classes
        self.layout_types = layout_types
        self.num_intent_classes = len(intent_classes)
        self.num_layout_types = len(layout_types)
        
        logger.info(f"Initializing KeyPilotVLM with base model: {config.base_model}")
        
        # Set up quantization config if using 4-bit or 8-bit loading
        quantization_config = None
        if config.load_in_4bit or config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.load_in_4bit,
                load_in_8bit=config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info(f"Using quantization: 4-bit={config.load_in_4bit}, 8-bit={config.load_in_8bit}")
        
        # Load base vision-language model
        try:
            self.base_model = AutoModelForVision2Seq.from_pretrained(
                config.base_model,
                quantization_config=quantization_config,
                device_map=config.device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            logger.info(f"Successfully loaded base model: {config.base_model}")
        except Exception as e:
            logger.error(f"Failed to load base model {config.base_model}: {e}")
            raise
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(
                config.base_model,
                trust_remote_code=True,
            )
            logger.info("Successfully loaded processor")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}")
            raise
        
        # Apply LoRA if enabled
        if config.use_lora:
            self._setup_lora()
        
        # Classification heads
        hidden_size = self.base_model.config.hidden_size
        self.intent_classifier = nn.Linear(hidden_size, self.num_intent_classes)
        self.layout_classifier = nn.Linear(hidden_size, self.num_layout_types)
        
        logger.info(f"Model initialized with {self.num_intent_classes} intent classes and {self.num_layout_types} layout types")
    
    def _setup_lora(self) -> None:
        """Set up LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("Setting up LoRA configuration")
        
        # Prepare model for k-bit training if using quantization
        if self.config.load_in_4bit or self.config.load_in_8bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA to model
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Image tensor [batch_size, channels, height, width]
            input_ids: Text input IDs [batch_size, sequence_length]
            attention_mask: Attention mask for input_ids
            labels: Optional dict with 'intent' and 'layout' labels for training
            
        Returns:
            Dictionary containing:
                - intent_logits: Intent classification logits
                - layout_logits: Layout classification logits
                - loss: Combined loss if labels provided
        """
        # Get base model outputs
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden state for classification
        # Use the last token's hidden state (or pool the sequence)
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        pooled_output = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Classification
        intent_logits = self.intent_classifier(pooled_output)
        layout_logits = self.layout_classifier(pooled_output)
        
        result = {
            'intent_logits': intent_logits,
            'layout_logits': layout_logits,
        }
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            
            if 'intent' not in labels or 'layout' not in labels:
                raise ValueError("labels must contain both 'intent' and 'layout' keys")
            
            intent_loss = loss_fct(intent_logits, labels['intent'])
            layout_loss = loss_fct(layout_logits, labels['layout'])
            
            # Combined loss (can be weighted if needed)
            total_loss = intent_loss + layout_loss
            
            result['loss'] = total_loss
            result['intent_loss'] = intent_loss
            result['layout_loss'] = layout_loss
        
        return result
    
    def predict(
        self,
        image: Image.Image,
        conversation_text: str,
    ) -> Tuple[str, str]:
        """
        Make prediction for a single input.
        
        Args:
            image: PIL Image of screen context
            conversation_text: Text of conversation history
            
        Returns:
            Tuple of (predicted_intent, predicted_layout)
            
        Raises:
            ValueError: If image or conversation_text is None
        """
        if image is None:
            raise ValueError("image cannot be None")
        if conversation_text is None:
            raise ValueError("conversation_text cannot be None")
        
        self.eval()
        with torch.no_grad():
            # Process inputs
            inputs = self.processor(
                text=conversation_text,
                images=image,
                return_tensors="pt",
            )
            
            # Move to model device
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = self.forward(
                pixel_values=inputs.get('pixel_values'),
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
            )
            
            # Get predictions
            intent_pred = torch.argmax(outputs['intent_logits'], dim=-1).item()
            layout_pred = torch.argmax(outputs['layout_logits'], dim=-1).item()
            
            return self.intent_classes[intent_pred], self.layout_types[layout_pred]
    
    def save_pretrained(self, output_dir: str) -> None:
        """
        Save model to directory.
        
        Args:
            output_dir: Directory to save model
            
        Raises:
            IOError: If model cannot be saved
        """
        try:
            logger.info(f"Saving model to {output_dir}")
            
            # Save base model (with LoRA if applicable)
            if self.config.use_lora:
                self.base_model.save_pretrained(output_dir)
            else:
                self.base_model.save_pretrained(output_dir)
            
            # Save processor
            self.processor.save_pretrained(output_dir)
            
            # Save classification heads
            torch.save({
                'intent_classifier': self.intent_classifier.state_dict(),
                'layout_classifier': self.layout_classifier.state_dict(),
                'intent_classes': self.intent_classes,
                'layout_types': self.layout_types,
            }, f"{output_dir}/classifiers.pt")
            
            logger.info(f"Model saved successfully to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError(f"Failed to save model: {e}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        config: ModelConfig,
    ) -> "KeyPilotVLM":
        """
        Load model from directory.
        
        Args:
            model_dir: Directory containing saved model
            config: Model configuration
            
        Returns:
            Loaded KeyPilotVLM instance
            
        Raises:
            FileNotFoundError: If model directory doesn't exist
            IOError: If model cannot be loaded
        """
        import os
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        try:
            logger.info(f"Loading model from {model_dir}")
            
            # Load classifier metadata
            classifier_path = f"{model_dir}/classifiers.pt"
            if not os.path.exists(classifier_path):
                raise FileNotFoundError(f"Classifier file not found: {classifier_path}")
            
            classifier_data = torch.load(classifier_path)
            intent_classes = classifier_data['intent_classes']
            layout_types = classifier_data['layout_types']
            
            # Create model instance
            model = cls(config, intent_classes, layout_types)
            
            # Load classifier weights
            model.intent_classifier.load_state_dict(classifier_data['intent_classifier'])
            model.layout_classifier.load_state_dict(classifier_data['layout_classifier'])
            
            logger.info(f"Model loaded successfully from {model_dir}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise IOError(f"Failed to load model: {e}")

