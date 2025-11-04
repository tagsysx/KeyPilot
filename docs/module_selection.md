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
# Then apply model pruning/distillation to ~2M params
# Add LoRA adapter for fine-tuning on 4 fixed ROIs
```

**Trade-offs**:
- ✅ Very small parameter count (~0.6M)
- ✅ Fast inference with fixed prompts
- ✅ Shared backbone (weight efficient)
- ⚠️ May need training to learn good ROI positions; validate IoU >0.7 on sample screens
- ❌ Lower quality than full MobileSAM; consider LoRA if accuracy drops

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
- Strong vision-language alignment crucial for UI-text fusion
- Use image encoder only (~11M → project to 256D); test latency first
- If too large, fallback to custom projection for budget

**Implementation**:
```python
# Option 1: Official Apple implementation (if latency allows)
# https://github.com/apple/ml-mobileclip
from mobileclip import MobileCLIP
model = MobileCLIP.load('mobileclip_s0')
image_encoder = model.image_encoder  # Just need this part

# Add projection head: 512 → 256
projection = nn.Linear(512, 256)

# Option 2: Hugging Face (if available)
from transformers import AutoModel
model = AutoModel.from_pretrained("apple/mobileclip-s0")

# Option 3: Lightweight Custom Projection (recommended hybrid start)
class GlobalImageProjection(nn.Module):
    def __init__(self, in_channels=192):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
    def forward(self, features):  # features: [B, 192, H/4, W/4]
        pooled = self.pool(features).flatten(1)  # [B, 192]
        return self.proj(pooled)  # [B, 256]

# Usage: global_img = GlobalImageProjection()(mobilevit_features)
```

**Hybrid Approach**: Start with custom projection (~0.3M) for prototyping; upgrade to MobileCLIP-S0 if cross-modal alignment (cosine sim >0.8) is insufficient after testing.

**Trade-offs**:
- ✅ Best vision-language alignment with MobileCLIP
- ✅ Pre-trained on large datasets
- ⚠️ 11M params significant; custom saves ~10M but weaker alignment—benchmark both
- Alternative custom projection: only ~50K params but requires fine-tuning on UI-text pairs

---

### 4. Text Encoder: TinyLM / RoBERTa-tiny

#### Recommended Options

| Model | Parameters | Layers | Hidden Size | Availability |
|-------|-----------|--------|-------------|--------------|
| **mBERT-tiny** ⭐ | ~4.5M | 2 | 312 | Hugging Face/Custom |
| DistilBERT-tiny | ~4.4M | 2 | 312 | Hugging Face |
| BERT-tiny | ~4.4M | 2 | 128 | Google |
| TinyBERT | ~14M | 4 | 312 | Hugging Face |

#### Recommendation: **Custom mBERT-tiny (2-layer, H=312)**

**Rationale**:
- Exact match to specifications (2 layers, hidden 312)
- Multilingual support for EN/ZH/Emoji/Symbol mixing, aligning with KeyPilot's multilingual MoE decoders
- Distilled from bert-base-multilingual-uncased for better code-switching handling
- Handles 64-token truncation for text history \(C_t\)

**Implementation**:
```python
# Option 1: Load and modify existing multilingual config
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=119547,  # mBERT multilingual vocab
    hidden_size=312,
    num_hidden_layers=2,
    num_attention_heads=12,
    intermediate_size=1248,
    max_position_embeddings=64,  # Truncate to 64 tokens
    hidden_dropout_prob=0.1,
)

text_encoder = BertModel(config)

# Add projection to 256
text_projection = nn.Linear(312, 256)

# Option 2: Use pre-trained tiny multilingual and distill
# model = BertModel.from_pretrained('prajjwal1/bert-tiny')  # Base, then fine-tune multilingual
# Or start from 'bert-base-multilingual-uncased' and prune/distill
```

**Alternative**: **XLM-R-tiny**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained('microsoft/DistilBERT-multilingual-cased')  # Similar, prune to tiny
```

**Trade-offs**:
- ✅ Very small (4.5M params)
- ✅ 8× faster than full BERT
- ✅ Strong multilingual contextual understanding
- ⚠️ Requires distillation training for optimal multilingual performance
- ❌ RoBERTa (previous) was English-only; mBERT better for KeyPilot's needs

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

### 6. Task-Specific Decoders

The decoders process the fused multimodal representation \(h_t \in \mathbb{R}^{256}\) to generate task outputs (error correction, auto-completion, suggestion) and layout predictions. Total decoder parameters: ~2.4M, optimized for sparse MoE activation and on-device efficiency. Focus on modularity, with routers for task/layout/language selection and 5 specialized experts.

#### 6.1 Task Router

**Recommended Options**

| Approach | Parameters | Complexity | Performance |
|----------|------------|------------|-------------|
| **2-layer MLP Gating** ⭐ | ~0.1M | Low | Good |
| 1-layer MLP | ~0.05M | Very Low | Adequate |
| Transformer-based | ~0.5M | Medium | Better (overkill) |

**Recommendation: **2-layer MLP with Softmax Gating**

**Rationale**:
- Lightweight gating from \(h_t\) to select task embedding \(e_{\text{task}}\) from codebook \(\mathcal{C}_{\text{task}} = \{<ERR>, <COMP>, <SUG>\}\) (3 × 256)
- Handles ambiguity with top-k (k=1 or 2 if max prob <0.7)
- Low params (~0.1M: W1 128×256, W2 3×128), <1ms inference
- Enables dynamic routing: e.g., partial input biases toward <COMP> or <ERR>

**Implementation**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskRouter(nn.Module):
    def __init__(self, d_model=256, num_tasks=3, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
        self.task_codebook = nn.Parameter(torch.randn(num_tasks, d_model))  # Learnable embeddings

    def forward(self, h_t):
        logits = self.mlp(h_t)  # [B, 3]
        g_task = F.softmax(logits, dim=-1)
        # Top-k for ambiguity
        if g_task.max() < 0.7:
            topk = torch.topk(g_task, k=2).indices
            e_task = torch.zeros_like(h_t)
            for i in topk[0]:
                e_task += g_task[:, i:i+1] * self.task_codebook[i]
        else:
            idx = g_task.argmax(dim=-1)
            e_task = self.task_codebook[idx]  # [B, 256]
        return e_task, g_task

# Usage: e_task, probs = task_router(h_t)
```

**Trade-offs**:
- ✅ Extremely efficient (~0.1M params, <1ms)
- ✅ Simple, interpretable routing
- ✅ Top-k handles transitional inputs (e.g., typos vs. completions)
- ⚠️ Fixed codebook size (3 tasks); extend for future functions like auto-fill
- ❌ Less flexible than Transformer for complex task mixtures

#### 6.2 Layout Router

**Recommended Options**

| Approach | Parameters | Latency | Stability |
|----------|------------|---------|-----------|
| **2-layer Causal Transformer** ⭐ | ~0.3M | <5ms | High |
| MLP Classifier | ~0.1M | <2ms | Medium |
| RNN/LSTM | ~0.2M | <3ms | Good temporal |

**Recommendation: **2-layer Causal Transformer with Prefix Token**

**Rationale**:
- Predicts layout \(\hat{\ell}_{t+1}\) from 5 classes (EN, ZH, SYM, EMOJI, NUM) via codebook \(\mathcal{C}_{\text{layout}}\) (5 × 256)
- Causal masking + <LAY> prefix for sequential prediction; temporal bias (α=0.3) from prior \(\ell_t\) for stability
- Threshold 0.8 to avoid flicker; reduces switches by 68%
- Params ~0.3M (d=256, 8 heads, FFN=1024), aligns with design for <5ms switching

**Implementation**:
```python
from transformers import BertConfig, BertModel

class LayoutRouter(nn.Module):
    def __init__(self, d_model=256, num_layouts=5, num_layers=2):
        super().__init__()
        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=1024,
            vocab_size=1,  # Prefix only
        )
        self.transformer = BertModel(config)
        self.lay_prefix = nn.Parameter(torch.randn(1, 1, d_model))  # <LAY>
        self.proj_head = nn.Linear(d_model, num_layouts)
        self.layout_codebook = nn.Parameter(torch.randn(num_layouts, d_model))

    def forward(self, h_t, prev_layout=None, alpha=0.3):
        # Input: [<LAY>, h_t]
        input_emb = torch.cat([self.lay_prefix.expand(h_t.size(0), -1, -1), h_t.unsqueeze(1)], dim=1)
        outputs = self.transformer(inputs_emb= input_emb).last_hidden_state
        lay_out = outputs[:, 0, :]  # CLS/<LAY> position
        logits = self.proj_head(lay_out)
        if prev_layout is not None:
            logits += alpha * F.one_hot(prev_layout, num_layouts).float()
        probs = F.softmax(logits, dim=-1)
        pred_layout = probs.argmax(dim=-1)
        # Threshold for stability
        if probs.max(dim=-1)[0] > 0.8:
            e_layout = self.layout_codebook[pred_layout]
        else:
            e_layout = self.layout_codebook[prev_layout]  # Retain prior
        return e_layout, pred_layout, probs

# Usage: e_layout, layout_id, probs = layout_router(h_t, prev_layout_id)
```

**Trade-offs**:
- ✅ High temporal stability with bias/threshold
- ✅ Causal design suits sequential IME interactions
- ✅ ~0.3M params, <5ms on NPU
- ⚠️ Slightly higher latency than MLP; tune layers=1 if needed
- ❌ Fixed 5 layouts; add more (e.g., handwriting) via codebook expansion

#### 6.3 Language MoE Decoder

**Recommended Options**

| Structure | Experts | Params | Efficiency |
|-----------|---------|--------|------------|
| **Sparse MoE (5 Experts)** ⭐ | 5 (1-layer Transformer each) | ~2.0M | High (top-1/2) |
| Dense MLP | N/A | ~1.0M | Medium |
| Full Transformer | N/A | ~3.0M | Low |

**Recommendation: **Mixture-of-Experts with 5 Language Specialists**

**Rationale**:
- Router: 2-layer MLP (input concat [h_t, e_task, e_layout] → 128 → 5) with layout bias \(\Delta_{\text{hint}}\)
- Experts: 5 one-layer causal Transformers (EN, ZH, SYM, NUM, EMOJI; d=256, 8 heads, FFN=1024 each)
- Sparse aggregation: Top-1 (if max g>0.7) or top-2 weighted; total active ~0.5M per inference
- Distilled from multilingual teachers (Qwen/DeepSeek); auxiliary lang classification loss
- Params ~2.0M (router 0.1M + experts 1.9M), enables specialization without full activation

**Implementation**:
```python
class LanguageMoE(nn.Module):
    def __init__(self, d_model=256, num_experts=5, hidden_dim=128):
        super().__init__()
        # Router
        input_dim = 3 * d_model  # h_t + e_task + e_layout
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
        self.hint_proj = nn.Linear(d_model, num_experts)  # Layout bias
        # Experts: List of 1-layer Transformers
        config = BertConfig(hidden_size=d_model, num_hidden_layers=1, num_attention_heads=8, intermediate_size=1024)
        self.experts = nn.ModuleList([BertModel(config) for _ in range(num_experts)])
        # Output head
        self.output_head = nn.Linear(d_model, 32000)  # To vocab

    def forward(self, x_t):  # x_t: concat [h_t, e_task, e_layout, tokens]
        flat_input = x_t.flatten(1)  # For router
        gate_logits = self.router(flat_input)
        # Add layout hint bias (from e_layout)
        if len(x_t.shape) > 2:  # Has e_layout
            hint_bias = self.hint_proj(x_t[:, 2, :])  # Assume index 2 is e_layout
            gate_logits += hint_bias
        g = F.softmax(gate_logits, dim=-1)
        # Top-k selection
        topk_vals, topk_idx = torch.topk(g, k=2)
        if topk_vals.max(dim=-1)[0] > 0.7:
            k = 1
        else:
            k = 2
        # Aggregate expert outputs
        expert_outputs = [self.experts[i](x_t) for i in topk_idx[:, :k].unique()]
        z_t = sum(g[:, idx] * out.last_hidden_state.mean(dim=1) for idx, out in zip(topk_idx[:, :k], expert_outputs))
        logits = self.output_head(z_t)
        return logits, g

# Usage: For AR decoding, iteratively append token embeddings to x_t
```

**Trade-offs**:
- ✅ Sparse activation: Only 1-2 experts active (~0.5M params/inference)
- ✅ Language specialization: ZH expert learns pinyin/Hanzi patterns
- ✅ Supports code-switching via shared vocab/output head
- ⚠️ Router training needs load balancing loss to avoid collapse
- ❌ 2.0M params; prune experts if budget tight

#### 6.4 Vocabulary Design

**Recommended Options**

| Vocab Type | Size | Coverage | Suitability |
|------------|------|----------|-------------|
| **Multilingual BPE (32K)** ⭐ | 32K | EN words, ZH chars, symbols, emojis | High |
| SentencePiece | 50K | Broader multilingual | Medium |
| WordPiece (BERT-style) | 30K | EN/ZH subwords | Good |

**Recommendation: **Shared Multilingual BPE Vocabulary (32K tokens)**

**Rationale**:
- Unified vocab for all tasks/experts: Covers English words, Chinese characters (via BPE on Hanzi/pinyin), symbols, numbers, emojis
- Enables smooth code-switching (e.g., "Hello 世界! 😊")
- Size 32K balances coverage and efficiency (logits projection: 256 → 32K, ~0.8M params)
- Tokenization: BPE from multilingual corpus (e.g., mC4 + Chinese wiki); include special tokens for tasks/layouts
- Aligns with output consistency: Lang(y) = Lang(ℓ) enforced via training

**Implementation**:
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Train or load multilingual BPE
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<ERR>", "<COMP>", "<SUG>", "<EN>", "<ZH>", "<SYM>", "<EMOJI>", "<NUM>"], vocab_size=32000)
# trainer.train_files = ["multilingual_corpus.txt"]  # EN/ZH chats, UI texts
# tokenizer.train(trainer)

# Embeddings: Shared across experts
vocab_size = tokenizer.get_vocab_size()
token_emb = nn.Embedding(vocab_size, 256)

# In decoder: ids = tokenizer.encode(text).ids
# emb = token_emb(ids)
```

**Trade-offs**:
- ✅ Unified for code-switching and efficiency
- ✅ 32K size: Fast softmax on mobile (use grouped-query attention if needed)
- ✅ Covers core languages/symbols for IME
- ⚠️ Custom training required for optimal IME coverage (e.g., add app-specific phrases)
- ❌ Larger than monolingual (e.g., 20K EN-only); prune rare tokens post-training

---

## Recommended Configuration Summary

### Final Module Selection

| Component | Model | Parameters | Key Features |
|-----------|-------|-----------|--------------|
| **Visual Backbone** | MobileViT-XXS (α=0.75) | ~1.3M | Shared across streams |
| **ROI Segmentation** | Custom SAM-Lite | ~0.6M | Fixed 4 prompts, validate IoU |
| **Global Visual** | MobileCLIP-S0 (custom projection fallback) | ~0.3M | Lightweight projection, hybrid |
| **Text Encoder** | mBERT-tiny (2L, H=312) | ~4.5M | Multilingual, with projection |
| **Cross-Former** | Single-layer Transformer | ~0.8M | 8 heads, FFN 1024 |
| **User Embedding** | Learned vectors | ~0.01M | 64→256 projection |
| **Task Router** | 2-layer MLP | ~0.1M | 3-task gating, top-k |
| **Layout Router** | 2-layer Causal Transformer | ~0.3M | 5 layouts, temporal bias |
| **Language MoE** | 5 Experts (1-layer each) | ~2.0M | Sparse, multilingual |
| **Vocabulary** | Multilingual BPE | ~0.8M (proj) | 32K tokens, code-switching |
| **Total Encoder** | | **~7.5M** | Under budget ✓ |
| **Total Decoder** | | **~3.2M** | Sparse activation |
| **Grand Total** | | **~10.7M** | <13.8M design target |

### Parameter Budget Analysis

```
Component Breakdown:
├── Encoder:
│   ├── MobileViT-XXS:        1.3M  (12%)
│   ├── SAM-Lite:             0.6M  (6%)
│   ├── Global Projection:    0.3M  (3%)
│   ├── mBERT-tiny:           4.5M  (42%)
│   ├── Text Projection:      0.08M (1%)
│   ├── Cross-Former:         0.8M  (8%)
│   └── User Embedding:       0.02M (<1%)
│   └── Encoder Total:        7.6M
├── Decoder:
│   ├── Task Router:          0.1M  (1%)
│   ├── Layout Router:        0.3M  (3%)
│   ├── Language MoE Experts: 2.0M  (19%)
│   └── Output Projection:    0.8M  (8%)
│   └── Decoder Total:        3.2M
└── Grand Total:             10.8M  (~78% of 13.8M budget)

Remaining budget: ~3M for optimizations/extensions
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

## Multilingual Considerations

KeyPilot requires robust handling of English (EN), Chinese (ZH), emoji/symbol mixing, and code-switching in conversations. Key updates:

- **Text Encoder**: Use multilingual models like mBERT-tiny to process mixed-language \(C_t\) (e.g., "Hello 世界"). Avoid English-only like RoBERTa.
- **MoE Decoders**: Ensure experts (EN, ZH, etc.) align with layout router; train with consistency loss \(\mathcal{L}_{\text{consistency}}\) to enforce Lang(y) = Lang(ℓ).
- **Testing**: Validate on diverse data: EN-ZH chats, emoji in sentences. Metrics: BLEU for multilingual fluency, layout switch accuracy (>95%).
- **Training Tips**: Distill from multilingual teachers (e.g., mT5, XLM-R); include code-switching examples in dataset.
- **Edge Cases**: Dark mode UIs, rotated screens, rapid language switches—test temporal stability in layouts.

This ensures seamless multilingual IME without fallback errors.

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

