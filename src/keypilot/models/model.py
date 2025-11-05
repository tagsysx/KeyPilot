"""
KeyPilot: Complete Vision-Language Typing Agent

Main model integrating encoder and decoder for intelligent IME.
Total parameters: ~10.7M (7.5M encoder + 3.2M decoder)
Target latency: <50ms end-to-end
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from .encoder import KeyPilotEncoder
from .decoder import KeyPilotDecoder


class KeyPilotVLM(nn.Module):
    """
    Complete KeyPilot Vision-Language Model.
    
    Functionality:
    - Error Detection & Correction
    - Auto-Completion
    - Proactive Suggestion
    - Adaptive Layout Switching
    """
    
    def __init__(self,
                 vocab_size: int = 32000,
                 d_model: int = 256,
                 num_tasks: int = 3,
                 num_layouts: int = 5,
                 num_experts: int = 5,
                 pretrained_backbone: bool = True,
                 user_emb_dim: int = 64):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Encoder: Vision-Language fusion
        self.encoder = KeyPilotEncoder(
            pretrained_backbone=pretrained_backbone,
            user_emb_dim=user_emb_dim,
            d_model=d_model
        )
        
        # Decoder: Task-specific outputs
        self.decoder = KeyPilotDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            num_tasks=num_tasks,
            num_layouts=num_layouts,
            num_experts=num_experts
        )
        
        # Task and layout mappings
        self.task_names = ['error_correction', 'auto_completion', 'suggestion']
        self.layout_names = ['EN', 'ZH', 'SYM', 'EMOJI', 'NUM']
    
    def forward(self,
                image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                user_id: Optional[torch.Tensor] = None,
                prev_layout: Optional[torch.Tensor] = None,
                prev_tokens: Optional[torch.Tensor] = None,
                return_aux: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of KeyPilot.
        
        Args:
            image: Screen image [B, 3, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            user_id: User ID [B] for personality
            prev_layout: Previous layout IDs [B] for temporal stability
            prev_tokens: Previous token IDs [B, N] for AR generation
            return_aux: Return auxiliary outputs for training
        
        Returns:
            outputs: Dictionary with predictions and metadata
        """
        # Encode multimodal input
        h_t, aux_outputs = self.encoder(
            image=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            user_id=user_id
        )
        
        # Decode to task outputs
        decoder_outputs = self.decoder(
            h_t=h_t,
            prev_layout=prev_layout,
            prev_tokens=prev_tokens
        )
        
        # Prepare outputs
        outputs = {
            'h_t': h_t,
            'task_probs': decoder_outputs['task_probs'],
            'layout_id': decoder_outputs['layout_id'],
            'layout_probs': decoder_outputs['layout_probs'],
            'logits': decoder_outputs['logits'],
            'expert_probs': decoder_outputs['expert_probs'],
        }
        
        if return_aux:
            outputs.update(aux_outputs)
            outputs['e_task'] = decoder_outputs['e_task']
            outputs['e_layout'] = decoder_outputs['e_layout']
        
        return outputs
    
    def predict(self,
                image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                user_id: Optional[torch.Tensor] = None,
                prev_layout: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                top_k: int = 50,
                top_p: float = 0.9) -> Dict[str, any]:
        """
        Inference mode: Get predictions with interpretable outputs.
        
        Args:
            image: Screen image [B, 3, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            user_id: User ID [B]
            prev_layout: Previous layout [B]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            predictions: Dictionary with human-readable outputs
        """
        self.eval()
        with torch.no_grad():
            # Encode
            h_t, _ = self.encoder(image, input_ids, attention_mask, user_id)
            
            # Determine task
            e_task, task_probs = self.decoder.task_router(h_t)
            task_id = task_probs.argmax(dim=-1)  # [B]
            
            # Determine layout
            e_layout, layout_id, layout_probs = self.decoder.layout_router(h_t, prev_layout)
            
            # Generate output based on task
            outputs = {}
            
            for i in range(h_t.size(0)):
                task = self.task_names[task_id[i].item()]
                layout = self.layout_names[layout_id[i].item()]
                
                if task == 'suggestion':
                    # Generate suggestion sequence
                    generated, metadata = self.decoder.generate(
                        h_t[i:i+1],
                        prev_layout[i:i+1] if prev_layout is not None else None,
                        max_length=5,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p
                    )
                    outputs[i] = {
                        'task': task,
                        'layout': layout,
                        'generated_tokens': generated[0].cpu().tolist(),
                        'task_confidence': task_probs[i].max().item(),
                        'layout_confidence': layout_probs[i].max().item()
                    }
                else:
                    # Single-step prediction (correction or completion)
                    logits, expert_probs = self.decoder.language_moe(
                        h_t[i:i+1], e_task[i:i+1], e_layout[i:i+1]
                    )
                    pred_token = logits.argmax(dim=-1)  # [1]
                    outputs[i] = {
                        'task': task,
                        'layout': layout,
                        'predicted_token': pred_token[0].item(),
                        'task_confidence': task_probs[i].max().item(),
                        'layout_confidence': layout_probs[i].max().item(),
                        'expert_distribution': expert_probs[i].cpu().tolist()
                    }
        
        return outputs
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get detailed model summary."""
        encoder_params = self.encoder.get_num_parameters()
        decoder_params = self.decoder.get_num_parameters()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'decoder_parameters': decoder_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # FP32
            'model_size_mb_int8': total_params / 1024 / 1024,  # INT8
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }
    
    @torch.jit.export
    def get_layout_name(self, layout_id: int) -> str:
        """Get layout name from ID."""
        if 0 <= layout_id < len(self.layout_names):
            return self.layout_names[layout_id]
        raise ValueError(f"Invalid layout_id: {layout_id}")
    
    @torch.jit.export
    def get_task_name(self, task_id: int) -> str:
        """Get task name from ID."""
        if 0 <= task_id < len(self.task_names):
            return self.task_names[task_id]
        raise ValueError(f"Invalid task_id: {task_id}")


# KeyPilotLoss has been moved to loss.py for better organization


def create_keypilot_model(config: Dict[str, any]) -> KeyPilotVLM:
    """
    Factory function to create KeyPilot model from config.
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        model: KeyPilotVLM instance
    """
    model = KeyPilotVLM(
        vocab_size=config.get('vocab_size', 32000),
        d_model=config.get('d_model', 256),
        num_tasks=config.get('num_tasks', 3),
        num_layouts=config.get('num_layouts', 5),
        num_experts=config.get('num_experts', 5),
        pretrained_backbone=config.get('pretrained_backbone', True),
        user_emb_dim=config.get('user_emb_dim', 64)
    )
    
    return model

