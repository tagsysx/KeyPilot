"""
Configuration management for KeyPilot project.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from loguru import logger


@dataclass
class ModelConfig:
    """Configuration for vision-language model."""
    
    base_model: str = "microsoft/phi-3-vision-128k-instruct"
    model_type: str = "vlm"  # vision-language model
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    device_map: str = "auto"
    max_length: int = 2048
    

@dataclass
class DataConfig:
    """Configuration for data processing and generation."""
    
    data_dir: str = "data"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None
    seed: int = 42
    
    # Data generation settings
    use_chatgpt: bool = True
    use_deepseek: bool = True
    num_agents: int = 4
    samples_per_agent: int = 1000
    

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    output_dir: str = "results"
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    
    # Optimizer settings
    optimizer: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Wandb settings
    use_wandb: bool = True
    wandb_project: str = "keypilot"
    wandb_run_name: Optional[str] = None
    

@dataclass
class DistillationConfig:
    """Configuration for model distillation to on-device models."""
    
    teacher_model_path: str = "models/finetuned/best_model"
    student_model: str = "microsoft/phi-2"
    distillation_alpha: float = 0.5
    distillation_temperature: float = 2.0
    output_dir: str = "models/distilled"
    

@dataclass
class KeyPilotConfig:
    """Main configuration class for KeyPilot project."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    
    # Input intent classes
    intent_classes: List[str] = field(default_factory=lambda: [
        "text", "symbol", "emoji", "numeric", "space", "enter", "delete"
    ])
    
    # Keyboard layout types
    layout_types: List[str] = field(default_factory=lambda: [
        "qwerty", "numeric", "symbol", "emoji"
    ])
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "KeyPilotConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            KeyPilotConfig instance
            
        Raises:
            FileNotFoundError: If config file does not exist
            ValueError: If config file is invalid
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            error_msg = f"Configuration file not found: {config_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if config_dict is None:
                error_msg = f"Configuration file is empty: {config_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            return cls(**config_dict)
            
        except yaml.YAMLError as e:
            error_msg = f"Invalid YAML in configuration file: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration file
            
        Raises:
            IOError: If file cannot be written
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = {
                'model': self.model.__dict__,
                'data': self.data.__dict__,
                'training': self.training.__dict__,
                'distillation': self.distillation.__dict__,
                'intent_classes': self.intent_classes,
                'layout_types': self.layout_types,
            }
            
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except IOError as e:
            error_msg = f"Failed to write configuration file: {e}"
            logger.error(error_msg)
            raise IOError(error_msg)

