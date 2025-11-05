"""
Loss functions for KeyPilot model training.

This module implements the comprehensive multitask loss function as defined in docs/method.md,
including encoder losses (alignment, contrastive) and decoder losses (text generation, task routing,
layout prediction, consistency, and MoE load balancing).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class KeyPilotLoss(nn.Module):
    """
    Multitask loss for KeyPilot training.

    Components:
    - Encoder losses: alignment, contrastive
    - Decoder losses: task CE, layout CE, text generation CE, consistency
    - Auxiliary losses: MoE load balancing

    Total Loss Formula:
        L_total = λ_enc × L_enc + λ_dec × L_dec

    Where:
        L_enc = λ_align × L_align + λ_contrastive × L_contrastive
        L_dec = L_text + λ_task × L_task + λ_layout × L_layout
                + λ_consistency × L_consistency + λ_load × L_load

    Loss Weights (default):
        - λ_enc = 0.3 (encoder loss weight)
        - λ_dec = 1.0 (decoder loss weight)
        - λ_align = 1.0 (cross-modal alignment)
        - λ_contrastive = 0.2 (ROI contrastive)
        - λ_task = 0.5 (task routing)
        - λ_layout = 0.4 (layout prediction)
        - λ_consistency = 0.3 (language-layout consistency)
        - λ_load = 0.01 (MoE load balancing)

    See docs/method.md for detailed documentation.
    """
    
    def __init__(self,
                 lambda_enc: float = 0.3,
                 lambda_dec: float = 1.0,
                 lambda_align: float = 1.0,
                 lambda_contrastive: float = 0.2,
                 lambda_task: float = 0.5,
                 lambda_layout: float = 0.4,
                 lambda_consistency: float = 0.3,
                 lambda_load: float = 0.01,
                 temperature: float = 0.07):
        """
        Initialize KeyPilot loss function.

        Args:
            lambda_enc: Weight for encoder loss
            lambda_dec: Weight for decoder loss
            lambda_align: Weight for cross-modal alignment loss
            lambda_contrastive: Weight for contrastive regularization
            lambda_task: Weight for task routing loss
            lambda_layout: Weight for layout prediction loss
            lambda_consistency: Weight for language-layout consistency
            lambda_load: Weight for MoE load balancing
            temperature: Temperature for contrastive learning (τ)
        """
        super().__init__()

        self.lambda_enc = lambda_enc
        self.lambda_dec = lambda_dec
        self.lambda_align = lambda_align
        self.lambda_contrastive = lambda_contrastive
        self.lambda_task = lambda_task
        self.lambda_layout = lambda_layout
        self.lambda_consistency = lambda_consistency
        self.lambda_load = lambda_load
        self.temperature = temperature
        
        self.ce_loss = nn.CrossEntropyLoss()
    
    def contrastive_loss(self, 
                         anchors: torch.Tensor, 
                         positives: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss for alignment.
        
        Formula:
            L = -1/B Σᵢ log[ exp(⟨vᵢ, tᵢ⟩/τ) / Σⱼ exp(⟨vᵢ, tⱼ⟩/τ) ]
        
        Args:
            anchors: [B, D] - Anchor embeddings (e.g., visual features)
            positives: [B, D] - Positive embeddings (e.g., text features)
        
        Returns:
            loss: Scalar contrastive loss
        """
        B = anchors.size(0)
        
        # L2 normalize
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(anchors, positives.T) / self.temperature  # [B, B]
        
        # Labels: diagonal entries are positive pairs
        labels = torch.arange(B, device=anchors.device)
        
        loss = self.ce_loss(logits, labels)
        return loss
    
    def load_balancing_loss(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Load balancing loss for MoE experts.
        
        Formula:
            L_load = Σᵢ (fᵢ - 1/N)²
        
        Where fᵢ is the average routing probability to expert i.
        Target: uniform distribution (1/N for N experts).
        
        Args:
            expert_probs: [B, num_experts] - Expert routing probabilities
        
        Returns:
            loss: Scalar load balancing loss
        """
        # Average usage across batch
        avg_usage = expert_probs.mean(dim=0)  # [num_experts]
        
        # Target: uniform distribution
        target = torch.ones_like(avg_usage) / avg_usage.size(0)
        
        # Quadratic loss
        loss = ((avg_usage - target) ** 2).sum()
        return loss
    
    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multitask loss.
        
        Args:
            outputs: Model outputs from forward pass containing:
                - 'global_feat': [B, 256] - Global image feature
                - 'text_feat': [B, 256] - Text CLS feature
                - 'roi_feats': [B, 4, 256] - ROI features (optional)
                - 'logits': [B, vocab_size] or [B, seq_len, vocab_size] - Token logits
                - 'task_probs': [B, 3] - Task routing probabilities
                - 'layout_probs': [B, 5] - Layout prediction logits
                - 'layout_id': [B] - Predicted layout ID
                - 'expert_probs': [B, 5] - MoE expert routing probabilities
            
            targets: Ground truth targets containing:
                - 'target_tokens': Token IDs for text generation
                - 'target_task': Task labels
                - 'target_layout': Layout labels
                - 'target_lang': Language labels (for consistency)
                - 'target_type': Token type labels
            

        Returns:
            losses: Dictionary of loss components:
                - 'loss_total': Total weighted loss
                - 'loss_encoder': Total encoder loss
                - 'loss_decoder': Total decoder loss
                - 'loss_align': Cross-modal alignment loss
                - 'loss_contrastive': Contrastive regularization loss
                - 'loss_text': Text generation loss
                - 'loss_task': Task routing loss
                - 'loss_layout': Layout prediction loss
                - 'loss_consistency': Language-layout consistency loss
                - 'loss_load': MoE load balancing loss
        
        Note: Distillation loss has been removed as it's not used in current training.
        """
        losses = {}
        
        # ============================================================
        # Encoder Losses
        # ============================================================
        
        if 'global_feat' in outputs and 'text_feat' in outputs:
            # Cross-modal alignment (InfoNCE)
            loss_align = self.contrastive_loss(outputs['global_feat'], outputs['text_feat'])
            losses['loss_align'] = loss_align
            
            # ROI contrastive regularization (if available)
            if 'roi_feats' in outputs:
                roi_mean = outputs['roi_feats'].mean(dim=1)  # [B, 256]
                loss_contrastive = self.contrastive_loss(roi_mean, outputs['text_feat'])
                losses['loss_contrastive'] = loss_contrastive
        
        # ============================================================
        # Decoder Losses
        # ============================================================
        
        # Text generation loss (cross-entropy)
        if 'logits' in outputs and 'target_tokens' in targets:
            logits = outputs['logits']
            target_tokens = targets['target_tokens']
            
            if logits.dim() == 3:  # [B, seq_len, vocab_size] - sequence generation
                logits_flat = logits.reshape(-1, logits.size(-1))
                target_flat = target_tokens.reshape(-1)
            else:  # [B, vocab_size] - single token generation
                logits_flat = logits
                if target_tokens.dim() == 2:
                    # Check if this is a single-token task (seq_len == 1) or multi-token task
                    if target_tokens.size(1) == 1:
                        target_flat = target_tokens.squeeze(1)  # [B] for single token
                    else:
                        # For single-token generation but multi-token targets, take first token
                        target_flat = target_tokens[:, 0]  # [B]
                else:
                    target_flat = target_tokens
            
            loss_text = self.ce_loss(logits_flat, target_flat)
            losses['loss_text'] = loss_text
        
        # Task routing loss
        if 'task_probs' in outputs and 'target_task' in targets:
            loss_task = self.ce_loss(outputs['task_probs'], targets['target_task'])
            losses['loss_task'] = loss_task
        
        # Layout prediction loss
        if 'layout_probs' in outputs and 'target_layout' in targets:
            loss_layout = self.ce_loss(outputs['layout_probs'], targets['target_layout'])
            losses['loss_layout'] = loss_layout
        
        # Language-layout consistency loss
        # Simplified implementation: penalize mismatch between predicted layout and target language
        # In practice, this encourages the layout predictor to be consistent with expected language
        if 'layout_probs' in outputs and 'target_lang' in targets:
            loss_consistency = F.cross_entropy(
                outputs['layout_probs'],
                targets['target_lang'],
                reduction='mean'
            )
            losses['loss_consistency'] = loss_consistency
        
        # MoE load balancing loss
        if 'expert_probs' in outputs:
            loss_load = self.load_balancing_loss(outputs['expert_probs'])
            losses['loss_load'] = loss_load
        
        
        # ============================================================
        # Aggregate Total Loss
        # ============================================================
        
        total_loss = 0.0
        
        # Encoder loss (weighted sum)
        enc_loss = 0.0
        if 'loss_align' in losses:
            enc_loss += self.lambda_align * losses['loss_align']
        if 'loss_contrastive' in losses:
            enc_loss += self.lambda_contrastive * losses['loss_contrastive']
        
        if enc_loss > 0:
            losses['loss_encoder'] = enc_loss
            total_loss += self.lambda_enc * enc_loss
        
        # Decoder loss (weighted sum)
        dec_loss = 0.0
        if 'loss_text' in losses:
            dec_loss += losses['loss_text']
        if 'loss_task' in losses:
            dec_loss += self.lambda_task * losses['loss_task']
        if 'loss_layout' in losses:
            dec_loss += self.lambda_layout * losses['loss_layout']
        if 'loss_consistency' in losses:
            dec_loss += self.lambda_consistency * losses['loss_consistency']
        if 'loss_load' in losses:
            dec_loss += self.lambda_load * losses['loss_load']
        
        if dec_loss > 0:
            losses['loss_decoder'] = dec_loss
            total_loss += self.lambda_dec * dec_loss
        
        
        losses['loss_total'] = total_loss
        
        return losses

