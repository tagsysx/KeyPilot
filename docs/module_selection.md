# KeyPilot Module Selection Guide

This document provides detailed recommendations for selecting the core modules of the Vision-Language Encoder.

## Overview

Based on the design specifications:
- **Target**: 13.8M parameters total
- **Latency**: ≤19 ms on mobile NPU
- **Quantization**: INT8 precision
- **Memory**: ~5.8 MB after quantization

## Module Recommendations

### 1. Visual Backbone: MobileViT

#### Recommended Options

| Model | Parameters | Input Size | Features | Availability |
|-------|-----------|------------|----------|--------------|
| **MobileViT-XXS** (α=0.75) ⭐ | ~1.3M | 256×256 | Best balance | `timm`, Custom |
| MobileViT-XS | ~2.3M | 256×256 | More capacity | `timm` |
| MobileNetV3-Small | ~2.5M | 224×224 | Faster, less capacity | `torchvision` |

#### Recommendation: **MobileViT-XXS (α=0.75)**

**Rationale**:
- Meets parameter budget (~1.3M)
- Combines CNN and Transformer for local+global features
- Output: 192 channels × H/4 × W/4 (as specified)
- Good NPU optimization potential

**Implementation Options**:
```python
# Option 1: Using timm (recommended)
import timm
model = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True)

# Option 2: Custom implementation with width multiplier
# from mobile_vit import MobileViT
# model = MobileViT(mode='xxs', width_multiplier=0.75)

# Option 3: Apple's official implementation
# https://github.com/apple/ml-cvnets
```

**Trade-offs**:
- ✅ Excellent balance of speed and accuracy
- ✅ Transformer attention for global context
- ✅ Well-suited for mobile NPUs
- ⚠️ Requires custom width multiplier (α=0.75)

---

### 2. Region Segmentation: MobileSAM

#### Recommended Options

| Model | Parameters | Speed | Mask Quality | Availability |
|-------|-----------|-------|--------------|--------------|
| **SAM-Lite** ⭐ | ~0.6M | Very Fast | Good | Custom |
| MobileSAM (ViT-Tiny) | ~5M | Fast | Excellent | GitHub |
| FastSAM | ~68M | Medium | Excellent | GitHub |

#### Recommendation: **Custom SAM-Lite**

**Rationale**:
- MobileSAM (5M params) is too large for our budget
- Need custom lightweight version with fixed prompts
- Only need 4 ROI masks (not general segmentation)

**Implementation Strategy**:
```python
# Lightweight U-Net style segmentation head
class SAMLite(nn.Module):
    def __init__(self, in_channels=192, num_masks=4):
        # Ultra-lightweight decoder
        # Input: Feature map from MobileViT (192 × H/4 × W/4)
        # Output: 4 binary masks
        
# Key optimizations:
# 1. Share MobileViT backbone (no extra encoder)
# 2. Fixed learnable prompt embeddings (4 × 256)
# 3. Depthwise-separable convolutions
# 4. Single-scale decoder (no multi-scale)
```

**Alternative**: If you need better quality:
```python
# Use MobileSAM with LoRA/pruning
from mobile_sam import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
# Then apply model pruning/distillation
```

**Trade-offs**:
- ✅ Very small parameter count (~0.6M)
- ✅ Fast inference with fixed prompts
- ✅ Shared backbone (weight efficient)
- ⚠️ May need training to learn good ROI positions
- ❌ Lower quality than full MobileSAM

---

### 3. Global Visual Features: MobileCLIP

#### Recommended Options

| Model | Parameters | Embedding Dim | Performance | Availability |
|-------|-----------|---------------|-------------|--------------|
| **MobileCLIP-S0** ⭐ | ~11M (image) | 512 → 256 | Good | Apple/GitHub |
| MobileCLIP-S1 | ~22M | 512 | Better | Apple/GitHub |
| CLIP-ViT-Tiny | ~5.5M | 192 | Baseline | OpenAI |
| SigLIP-Small | ~12M | 384 | Competitive | Google/HF |

#### Recommendation: **MobileCLIP-S0 (image encoder only)**

**Rationale**:
- Specifically designed for mobile deployment
- Strong vision-language alignment
- Can use just the image encoder (~11M → project to 256D)

**Implementation**:
```python
# Option 1: Official Apple implementation
# https://github.com/apple/ml-mobileclip
from mobileclip import MobileCLIP
model = MobileCLIP.load('mobileclip_s0')
image_encoder = model.image_encoder  # Just need this part

# Option 2: Hugging Face (if available)
from transformers import AutoModel
model = AutoModel.from_pretrained("apple/mobileclip-s0")

# Add projection head: 512 → 256
projection = nn.Linear(512, 256)
```

**Alternative**: **Lightweight Custom Projection**
```python
# If 11M is too large, use simpler approach:
# Global average pooling on MobileViT features + MLP
class GlobalImageProjection(nn.Module):
    def __init__(self):
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(192, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
```

**Trade-offs**:
- ✅ Best vision-language alignment
- ✅ Pre-trained on large datasets
- ⚠️ 11M params is significant (but manageable)
- Alternative custom projection: only ~50K params but weaker alignment

---

### 4. Text Encoder: TinyLM / RoBERTa-tiny

#### Recommended Options

| Model | Parameters | Layers | Hidden Size | Availability |
|-------|-----------|--------|-------------|--------------|
| **RoBERTa-tiny** ⭐ | ~4.5M | 2 | 312 | Custom/HF |
| DistilBERT-tiny | ~4.4M | 2 | 312 | Hugging Face |
| BERT-tiny | ~4.4M | 2 | 128 | Google |
| TinyBERT | ~14M | 4 | 312 | Hugging Face |

#### Recommendation: **Custom RoBERTa-tiny (2-layer, H=312)**

**Rationale**:
- Exact match to specifications (2 layers, hidden 312)
- RoBERTa better than BERT for our use case
- Can distill from larger RoBERTa model

**Implementation**:
```python
# Option 1: Load and modify existing model
from transformers import RobertaConfig, RobertaModel

config = RobertaConfig(
    vocab_size=50265,  # RoBERTa vocab
    hidden_size=312,
    num_hidden_layers=2,
    num_attention_heads=12,
    intermediate_size=1248,
    max_position_embeddings=64,  # Truncate to 64 tokens
    hidden_dropout_prob=0.1,
)

text_encoder = RobertaModel(config)

# Add projection to 256
text_projection = nn.Linear(312, 256)

# Option 2: Use pre-trained tiny model and distill
model = RobertaModel.from_pretrained('prajjwal1/bert-tiny')
# Fine-tune with distillation
```

**Alternative**: **MiniLM**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('microsoft/MiniLM-L6-H384-uncased')
# Then prune to 2 layers
```

**Trade-offs**:
- ✅ Very small (4.5M params)
- ✅ 8× faster than full BERT
- ✅ Good contextual understanding
- ⚠️ May need distillation training for best performance

---

### 5. Cross-Modal Fusion: Cross-Former

#### Recommended Options

| Approach | Parameters | Complexity | Performance |
|----------|-----------|------------|-------------|
| **Single-layer Transformer** ⭐ | ~0.8M | Low | Good |
| Cross-Attention Layer | ~0.5M | Low | Adequate |
| Two-layer Transformer | ~1.6M | Medium | Better |

#### Recommendation: **Single-layer Transformer with FlashAttention**

**Specifications** (as per design):
- 1 layer
- 8 attention heads
- d_model = 256
- FFN dimension = 1024
- Gated cross-attention
- FlashAttention-2

**Implementation**:
```python
from transformers import BertConfig, BertLayer

config = BertConfig(
    hidden_size=256,
    num_attention_heads=8,
    intermediate_size=1024,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
)

cross_former = BertLayer(config)

# For FlashAttention-2 (optional, for training efficiency)
# pip install flash-attn
from flash_attn import flash_attn_func
```

**Trade-offs**:
- ✅ Simple and efficient
- ✅ ~0.8M parameters
- ✅ FlashAttention reduces memory
- ✅ Sufficient for 9-token sequence

---

## Recommended Configuration Summary

### Final Module Selection

| Component | Model | Parameters | Key Features |
|-----------|-------|-----------|--------------|
| **Visual Backbone** | MobileViT-XXS (α=0.75) | ~1.3M | Shared across streams |
| **ROI Segmentation** | Custom SAM-Lite | ~0.6M | Fixed 4 prompts |
| **Global Visual** | MobileCLIP-S0 (custom projection) | ~0.3M | Lightweight projection |
| **Text Encoder** | RoBERTa-tiny (2L, H=312) | ~4.5M | With projection |
| **Cross-Former** | Single-layer Transformer | ~0.8M | 8 heads, FFN 1024 |
| **User Embedding** | Learned vectors | ~0.01M | 64→256 projection |
| **Total Encoder** | | **~7.5M** | Under budget ✓ |

### Parameter Budget Analysis

```
Component Breakdown:
├── MobileViT-XXS:        1.3M  (17%)
├── SAM-Lite:             0.6M  (8%)
├── Global Projection:    0.3M  (4%)
├── RoBERTa-tiny:         4.5M  (60%)
├── Text Projection:      0.08M (1%)
├── Cross-Former:         0.8M  (11%)
├── User Embedding:       0.02M (<1%)
└── Total:                7.6M  (~55% of 13.8M budget)

Remaining budget: ~6M parameters for decoder
```

## Implementation Priority

### Phase 1: Core Components (Week 1-2)
1. ✅ MobileViT-XXS backbone
2. ✅ RoBERTa-tiny text encoder
3. ✅ Cross-Former fusion layer
4. ✅ Basic forward pass

### Phase 2: Visual Enhancement (Week 3)
5. ✅ SAM-Lite ROI extraction
6. ✅ Global visual projection (start simple, upgrade to MobileCLIP later)

### Phase 3: Optimization (Week 4)
7. ✅ INT8 quantization
8. ✅ FlashAttention integration
9. ✅ Mobile NPU optimization

## Alternative Lightweight Configuration

If you need even smaller model:

| Component | Lightweight Option | Parameters |
|-----------|-------------------|------------|
| Visual Backbone | MobileNetV3-Small | ~2.5M |
| ROI Segmentation | Fixed grid (no model) | 0M |
| Global Visual | Simple pooling + MLP | ~0.05M |
| Text Encoder | BERT-tiny (L=2, H=128) | ~4M |
| Cross-Former | Cross-attention only | ~0.4M |
| **Total** | | **~7M** |

## Pretrained Weights Sources

### 1. MobileViT
- `timm` library: `pip install timm`
- Apple ML-CVNets: https://github.com/apple/ml-cvnets
- ImageNet pretrained available

### 2. MobileCLIP
- Official: https://github.com/apple/ml-mobileclip
- Hugging Face (community): Check model hub

### 3. RoBERTa-tiny
- Hugging Face: `prajjwal1/bert-tiny` (similar architecture)
- Custom distillation from `roberta-base`

### 4. MobileSAM
- Official: https://github.com/ChaoningZhang/MobileSAM
- Need to create lightweight version

## Next Steps

1. **Validate module choices** with your requirements
2. **Set up environment** with required libraries
3. **Implement base encoder** with selected modules
4. **Test inference speed** on target device
5. **Iterate and optimize** based on benchmarks

## Questions to Consider

1. **Do you have access to mobile NPU** for testing? (affects optimization strategy)
2. **What's your training data size?** (affects whether to use pretrained vs. train from scratch)
3. **Do you need multilingual support** in text encoder? (RoBERTa is English-only)
4. **What's your deployment target?** (iOS/Android affects quantization approach)
5. **Do you prefer simplicity or performance?** (affects MobileCLIP vs simple projection choice)

---

**Recommendation**: Start with the main configuration (7.5M params) and optimize based on real-world performance. The remaining 6M parameter budget gives flexibility for the decoder.

