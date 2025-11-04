# Quick Decision Guide for Module Selection

**TL;DR**: Recommended choices for getting started quickly.

## Recommended Configuration

### Option A: Balanced (Recommended) ⭐

Best balance between performance and simplicity.

```python
modules = {
    "visual_backbone": "timm/mobilevit_xxs",      # 1.3M params
    "roi_segmentation": "custom_sam_lite",         # 0.6M params
    "global_visual": "simple_projection",          # 0.05M params
    "text_encoder": "prajjwal1/bert-tiny",        # 4.5M params
    "cross_former": "single_layer_transformer",    # 0.8M params
}
# Total: ~7.2M parameters
```

**Pros**:
- Easy to implement (all components available)
- Fast training and inference
- Good starting point for iteration

**Cons**:
- Simple projection may be weaker than MobileCLIP
- May need fine-tuning for best performance

### Option B: High Performance

For best accuracy, willing to use more parameters.

```python
modules = {
    "visual_backbone": "timm/mobilevit_xxs",           # 1.3M params
    "roi_segmentation": "mobilesam_pruned",            # 2.0M params
    "global_visual": "apple/mobileclip-s0",            # 11M params (!)
    "text_encoder": "sentence-transformers/MiniLM",    # 22M params (!)
    "cross_former": "two_layer_transformer",           # 1.6M params
}
# Total: ~38M parameters (needs aggressive pruning/distillation)
```

**Pros**:
- Best vision-language understanding
- Strong pretrained features
- Better generalization

**Cons**:
- Exceeds parameter budget significantly
- Requires distillation/pruning
- Slower inference

### Option C: Ultra-Lightweight

For extreme efficiency or very constrained devices.

```python
modules = {
    "visual_backbone": "mobilenetv3_small",      # 2.5M params
    "roi_segmentation": "fixed_grid",            # 0M params (no model)
    "global_visual": "avg_pool_mlp",            # 0.05M params
    "text_encoder": "custom_lstm",              # 2.0M params
    "cross_former": "cross_attention_only",     # 0.4M params
}
# Total: ~5M parameters
```

**Pros**:
- Extremely fast
- Very small memory footprint
- Easy mobile deployment

**Cons**:
- Lower accuracy
- Less sophisticated features
- May struggle with complex scenarios

## Decision Matrix

| Factor | Option A | Option B | Option C |
|--------|----------|----------|----------|
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Parameter Budget** | ✅ Within | ❌ Exceeds 3× | ✅ Well under |
| **Implementation Time** | 1-2 weeks | 4-6 weeks | 1 week |

## My Recommendation

**Start with Option A**, then iterate:

### Phase 1 (Week 1-2): Get it working
- Use Option A configuration
- Focus on correct implementation
- Validate on simple examples
- Measure baseline performance

### Phase 2 (Week 3-4): Optimize selectively
- If visual understanding is weak → Upgrade to MobileCLIP
- If text understanding is weak → Try RoBERTa-tiny
- If ROI quality is poor → Upgrade to MobileSAM
- Keep what works, upgrade what doesn't

### Phase 3 (Week 5+): Production optimization
- Apply quantization (INT8)
- Distill from better models if needed
- Profile and optimize bottlenecks
- Target device testing

## Component-Specific Recommendations

### 1. Visual Backbone: **MobileViT-XXS**
```bash
pip install timm
```
```python
import timm
backbone = timm.create_model('mobilevit_xxs', pretrained=True, features_only=True)
```
**Why**: Best balance, available in timm, pretrained weights available.

### 2. ROI Segmentation: **Start Simple**

**Week 1 approach** (no model):
```python
# Fixed grid approach
def get_rois(image):
    h, w = image.shape[-2:]
    return [
        image[:, :, 0:h//2, :],        # Top half (title/header)
        image[:, :, h//2:, w//4:3*w//4],  # Center (input field)
        image[:, :, h//2:, :],         # Bottom half (keyboard)
        image[:, :, :h//4, :],         # Top bar (status)
    ]
```

**Week 3 upgrade** (if needed):
```python
# Custom SAM-Lite with learnable prompts
class SAMLite(nn.Module):
    # Lightweight segmentation head
```

### 3. Global Visual: **Simple Projection First**
```python
class GlobalProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(192, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
    
    def forward(self, features):
        x = self.pool(features).flatten(1)
        return self.proj(x)
```

### 4. Text Encoder: **Use Pretrained Small Model**
```python
from transformers import AutoModel, AutoTokenizer

# Option 1: BERT-tiny (easiest)
model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')

# Option 2: Custom RoBERTa-tiny (if you need RoBERTa specifically)
from transformers import RobertaConfig, RobertaModel
config = RobertaConfig(
    hidden_size=312,
    num_hidden_layers=2,
    num_attention_heads=12,
    max_position_embeddings=64,
)
model = RobertaModel(config)
# Then distill from roberta-base
```

### 5. Cross-Former: **Use Standard Transformer Layer**
```python
from torch.nn import TransformerEncoderLayer, TransformerEncoder

encoder_layer = TransformerEncoderLayer(
    d_model=256,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.1,
    batch_first=True,
)
cross_former = TransformerEncoder(encoder_layer, num_layers=1)
```

## Installation Commands

```bash
# Core dependencies
pip install torch torchvision
pip install transformers
pip install timm
pip install einops

# Optional (for optimization)
pip install flash-attn  # For FlashAttention-2
pip install onnx onnxruntime  # For model export
pip install optimum  # For quantization

# For training
pip install wandb  # Experiment tracking
pip install tqdm  # Progress bars
```

## Validation Checklist

Before committing to a configuration, validate:

- [ ] Can load all models without errors
- [ ] Forward pass produces correct output shapes
- [ ] Total parameters < 13.8M
- [ ] Inference time < 19ms (on target device or simulated)
- [ ] Memory usage < 6MB after quantization
- [ ] Pretrained weights available or can train from scratch

## Questions to Ask

1. **Do you have training data ready?**
   - Yes → Can use simpler models and train end-to-end
   - No → Need strong pretrained models (Option B)

2. **Can you test on target device?**
   - Yes → Benchmark each option and choose best
   - No → Start with Option A (most balanced)

3. **How much time do you have?**
   - 1-2 weeks → Option A
   - 4+ weeks → Option B with optimization
   - < 1 week → Option C

4. **What's more important?**
   - Speed → Option C or A
   - Accuracy → Option B
   - Balance → Option A

---

## Final Recommendation

**Use Option A (Balanced) with this specific setup**:

```python
# encoder_config.py
ENCODER_CONFIG = {
    # Visual
    "backbone": "mobilevit_xxs",
    "backbone_pretrained": True,
    "roi_method": "fixed_grid",  # Start simple
    "global_projection": "simple_mlp",
    
    # Text
    "text_model": "prajjwal1/bert-tiny",
    "text_max_length": 64,
    "text_projection_dim": 256,
    
    # Fusion
    "fusion_layers": 1,
    "fusion_heads": 8,
    "fusion_ffn_dim": 1024,
    
    # User embedding
    "user_embed_dim": 64,
    "user_proj_dim": 256,
}
```

This gives you:
- ✅ Quick start (all components readily available)
- ✅ Within parameter budget
- ✅ Good baseline performance
- ✅ Easy to upgrade components later
- ✅ Clear upgrade path to Option B if needed

**Next step**: Shall I implement the encoder with this configuration?

