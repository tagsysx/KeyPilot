"""
Vision-Language Encoder for KeyPilot

This module implements the multimodal encoder that jointly processes:
- Screen images (I_t) via MobileViT backbone
- Text context (C_t) via mBERT-tiny
- User personality embedding (P_u)

Total parameters: ~7.5M
Target latency: ≤19ms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from transformers import BertConfig, BertModel
import timm


class MobileViTBackbone(nn.Module):
    """
    MobileViT-XXS (α=0.75) visual backbone for shared feature extraction.
    
    Output: 64 channels × H/4 × W/4
    Parameters: ~1.3M
    """
    
    def __init__(self, pretrained: bool = True, width_multiplier: float = 0.75):
        super().__init__()
        # Using timm for MobileViT
        self.backbone = timm.create_model(
            'mobilevit_xxs', 
            pretrained=pretrained, 
            features_only=True,
            out_indices=[3]  # Get feature map at 4x downsampling
        )
        self.width_multiplier = width_multiplier
        
        # Output channels: 64 (actual output from mobilevit_xxs)
        self.out_channels = 64
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image [B, 3, H, W], typically [B, 3, 512, 256]
        
        Returns:
            features: [B, 64, H/4, W/4]
        """
        features = self.backbone(x)
        if isinstance(features, list):
            features = features[-1]  # Get last feature map
        return features


class SAMLite(nn.Module):
    """
    Lightweight ROI segmentation module for 4 fixed regions:
    - Input field
    - Chat bubble
    - Keyboard area
    - Title bar
    
    Parameters: ~0.6M
    """
    
    def __init__(self, in_channels: int = 64, num_masks: int = 4):
        super().__init__()
        self.num_masks = num_masks
        self.in_channels = in_channels
        
        # Projection layer to transform features to 256D
        self.roi_proj = nn.Linear(in_channels, 256)
        
        # Lightweight U-Net style decoder with depthwise-separable convs
        self.decoder = nn.Sequential(
            # Upsampling path
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(128, 64, 3, padding=1, groups=64),  # Depthwise
            nn.Conv2d(64, 64, 1),  # Pointwise
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Output to num_masks binary masks
            nn.Conv2d(64, num_masks, 1)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: MobileViT features [B, 64, H/4, W/4]
        
        Returns:
            masks: Binary masks [B, num_masks, H, W]
            roi_features: ROI-pooled features [B, num_masks, 256]
        """
        # Generate masks
        masks = self.decoder(features)
        masks = torch.sigmoid(masks)  # Binary masks
        
        # ROI pooling: extract features for each mask region
        B, C, H, W = features.shape
        roi_features = []
        
        # Upsample masks to match feature map size for pooling
        masks_resized = F.interpolate(masks, size=(H, W), mode='bilinear', align_corners=False)
        
        for i in range(self.num_masks):
            mask = masks_resized[:, i:i+1, :, :]  # [B, 1, H, W]
            # Masked average pooling
            masked_features = features * mask
            pooled = masked_features.sum(dim=[2, 3]) / (mask.sum(dim=[2, 3]) + 1e-6)  # [B, in_channels]
            # Project to 256D
            roi_feat = self.roi_proj(pooled)  # [B, 256]
            roi_features.append(roi_feat)
        
        roi_features = torch.stack(roi_features, dim=1)  # [B, num_masks, 256]
        
        return masks, roi_features


class GlobalImageProjection(nn.Module):
    """
    Lightweight global image feature projection.
    Alternative to MobileCLIP for smaller footprint.
    
    Parameters: ~0.3M
    """
    
    def __init__(self, in_channels: int = 64, out_dim: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: MobileViT features [B, 64, H/4, W/4]
        
        Returns:
            global_feat: [B, 256]
        """
        pooled = self.pool(features).flatten(1)  # [B, 64]
        return self.proj(pooled)  # [B, 256]


class TextEncoder(nn.Module):
    """
    Multilingual text encoder using mBERT-tiny (2 layers, H=312).
    
    Parameters: ~4.5M
    """
    
    def __init__(self, hidden_size: int = 312, out_dim: int = 256):
        super().__init__()
        
        # Custom mBERT-tiny configuration
        config = BertConfig(
            vocab_size=119547,  # mBERT multilingual vocab
            hidden_size=hidden_size,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=1248,
            max_position_embeddings=64,  # Truncate to 64 tokens
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.encoder = BertModel(config)
        self.projection = nn.Linear(hidden_size, out_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [B, seq_len], max_len=64
            attention_mask: Attention mask [B, seq_len]
        
        Returns:
            text_feat: [CLS] token embedding [B, 256]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, 312]
        return self.projection(cls_token)  # [B, 256]


class CrossFormer(nn.Module):
    """
    Single-layer Transformer for cross-modal fusion.
    
    Fuses: [CLS] + [IMG] + 4×[ROI] + [SEP] + [CLS_text] + [P]
    Total: 9 tokens
    
    Parameters: ~0.8M
    """
    
    def __init__(self, d_model: int = 256, num_heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        
        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            intermediate_size=ffn_dim,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.transformer = BertModel(config, add_pooling_layer=False)
        
        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, 
                img_feat: torch.Tensor,
                roi_feats: torch.Tensor,
                text_feat: torch.Tensor,
                user_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: Global image feature [B, 256]
            roi_feats: ROI features [B, 4, 256]
            text_feat: Text CLS feature [B, 256]
            user_feat: User personality [B, 256]
        
        Returns:
            h_t: Fused representation [B, 256]
        """
        B = img_feat.size(0)
        
        # Prepare sequence: [CLS] || [IMG] || [ROI1-4] || [SEP] || [CLS_text] || [P]
        cls = self.cls_token.expand(B, -1, -1)
        sep = self.sep_token.expand(B, -1, -1)
        
        sequence = torch.cat([
            cls,                                   # [B, 1, 256]
            img_feat.unsqueeze(1),                 # [B, 1, 256]
            roi_feats,                             # [B, 4, 256]
            sep,                                   # [B, 1, 256]
            text_feat.unsqueeze(1),                # [B, 1, 256]
            user_feat.unsqueeze(1)                 # [B, 1, 256]
        ], dim=1)  # [B, 9, 256]
        
        # Process through transformer
        outputs = self.transformer(inputs_embeds=sequence)
        
        # Extract CLS position output as h_t
        h_t = outputs.last_hidden_state[:, 0, :]  # [B, 256]
        
        return h_t


class KeyPilotEncoder(nn.Module):
    """
    Complete Vision-Language Encoder for KeyPilot.
    
    Total parameters: ~7.5M
    Target latency: ≤19ms
    """
    
    def __init__(self, 
                 pretrained_backbone: bool = True,
                 user_emb_dim: int = 64,
                 d_model: int = 256):
        super().__init__()
        
        # Visual components
        self.backbone = MobileViTBackbone(pretrained=pretrained_backbone)
        self.sam_lite = SAMLite(in_channels=64, num_masks=4)
        self.global_proj = GlobalImageProjection(in_channels=64, out_dim=d_model)
        
        # Text encoder
        self.text_encoder = TextEncoder(hidden_size=312, out_dim=d_model)
        
        # User personality embedding
        self.user_embedding = nn.Embedding(1000, user_emb_dim)  # Support 1000 users
        self.user_proj = nn.Linear(user_emb_dim, d_model)
        
        # Cross-modal fusion
        self.cross_former = CrossFormer(d_model=d_model, num_heads=8, ffn_dim=1024)
    
    def forward(self,
                image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                user_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of the encoder.
        
        Args:
            image: Screen image [B, 3, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            user_id: User ID [B] for personality embedding
        
        Returns:
            h_t: Multimodal representation [B, 256]
            aux_outputs: Auxiliary outputs (masks, features) for training
        """
        B = image.size(0)
        
        # Visual encoding
        visual_features = self.backbone(image)  # [B, 192, H/4, W/4]
        
        # Dual-path visual processing
        global_feat = self.global_proj(visual_features)  # [B, 256]
        masks, roi_feats = self.sam_lite(visual_features)  # masks: [B, 4, H, W], roi: [B, 4, 256]
        
        # Text encoding
        text_feat = self.text_encoder(input_ids, attention_mask)  # [B, 256]
        
        # User personality
        if user_id is None:
            user_id = torch.zeros(B, dtype=torch.long, device=image.device)
        user_emb = self.user_embedding(user_id)  # [B, 64]
        user_feat = self.user_proj(user_emb)  # [B, 256]
        
        # Cross-modal fusion
        h_t = self.cross_former(global_feat, roi_feats, text_feat, user_feat)  # [B, 256]
        
        # Auxiliary outputs for training losses
        aux_outputs = {
            'visual_features': visual_features,
            'global_feat': global_feat,
            'roi_feats': roi_feats,
            'roi_masks': masks,
            'text_feat': text_feat,
            'user_feat': user_feat
        }
        
        return h_t, aux_outputs
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'backbone': sum(p.numel() for p in self.backbone.parameters()),
            'sam_lite': sum(p.numel() for p in self.sam_lite.parameters()),
            'global_proj': sum(p.numel() for p in self.global_proj.parameters()),
            'text_encoder': sum(p.numel() for p in self.text_encoder.parameters()),
            'user_embedding': sum(p.numel() for p in self.user_embedding.parameters()),
            'user_proj': sum(p.numel() for p in self.user_proj.parameters()),
            'cross_former': sum(p.numel() for p in self.cross_former.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }

