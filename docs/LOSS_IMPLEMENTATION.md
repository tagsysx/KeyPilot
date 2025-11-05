# KeyPilot Loss Function Implementation

## Overview

The KeyPilot training loss is a comprehensive multitask objective that jointly optimizes the vision-language encoder and task-specific decoders. The implementation in `src/keypilot/models/keypilot_model.py` follows the specifications in `docs/method.md`.

## Total Loss Formula

```
L_total = λ_enc × L_enc + λ_dec × L_dec
```

Where:
- λ_enc = 0.3 (encoder loss weight)
- λ_dec = 1.0 (decoder loss weight)

## 1. Encoder Loss (`L_enc`)

### Formula
```
L_enc = λ_align × L_align + λ_contrastive × L_contrastive
```

Where:
- λ_align = 1.0
- λ_contrastive = 0.2

### 1.1 Cross-Modal Alignment Loss (`L_align`)

**Purpose**: Align global visual and textual embeddings using InfoNCE contrastive learning.

**Formula**:
```
L_align = -1/B Σᵢ log[ exp(⟨vᵢ, tᵢ⟩/τ) / Σⱼ exp(⟨vᵢ, tⱼ⟩/τ) ]
```

Where:
- `v_i` = Global image feature `[IMG]_i` ∈ ℝ^256 (L2-normalized)
- `t_i` = Text CLS feature `[CLS_text]_i` ∈ ℝ^256 (L2-normalized)
- τ = 0.07 (temperature parameter)
- B = batch size

**Implementation**: `KeyPilotLoss.contrastive_loss()`

**Behavior**:
- Encourages matched vision-text pairs to be close in embedding space
- Pushes apart negative pairs from the same batch
- Helps model learn UI-text correspondences (e.g., chat bubbles → conversational intent)

### 1.2 Contrastive Regularization Loss (`L_contrastive`)

**Purpose**: Refine local feature alignment between ROI tokens and textual embeddings.

**Formula**:
```
L_contrastive = -1/B Σᵢ log[ exp(⟨r̄ᵢ, tᵢ⟩/τ) / Σₖ exp(⟨r̄ₖ, tₖ⟩/τ) ]
```

Where:
- `r̄_i` = Average of 4 ROI tokens: `1/4 Σₘ [ROI_m]_i` ∈ ℝ^256
- ROIs represent: input field, chat bubble, keyboard, title bar

**Behavior**:
- Promotes spatial awareness
- Associates specific UI regions with contextual intent
- Improves robustness to layout variations

## 2. Decoder Loss (`L_dec`)

### Formula
```
L_dec = L_text + λ_task × L_task + λ_layout × L_layout 
        + λ_consistency × L_consistency + λ_load × L_load
```

Where:
- λ_task = 0.5
- λ_layout = 0.4
- λ_consistency = 0.3
- λ_load = 0.01

### 2.1 Text Generation Loss (`L_text`)

**Purpose**: Unified cross-entropy loss for all text generation tasks.

**Formula**:
```
L_text = -1/|Y| Σᵢ log p(yᵢ | hₜ, e_task, y₍<i₎; θ)
```

Where:
- |Y| = target sequence length
  - Error Correction: |Y| = 1 (single token)
  - Auto-Completion: |Y| = 1 (single token)
  - Suggestion: |Y| = 5 (multi-token sequence)
- Vocabulary size: |V| = 32,000 (multilingual BPE)

**Implementation**: Standard cross-entropy over predicted logits

### 2.2 Task Routing Loss (`L_task`)

**Purpose**: Supervise the task gating distribution for dynamic task selection.

**Formula**:
```
L_task = -1/B Σᵦ Σᵢ yᵢ⁽ᵇ⁾ log g_task,i⁽ᵇ⁾
```

Where:
- y_i = One-hot pseudo-label for task i ∈ {ERROR, COMPLETION, SUGGESTION}
- g_task = Task routing probabilities from gating network
- Pseudo-labels derived from user actions:
  - Backspace → ERROR
  - Continued typing → COMPLETION
  - Suggestion acceptance → SUGGESTION

### 2.3 Layout Prediction Loss (`L_layout`)

**Purpose**: Train the layout decoder to predict keyboard mode.

**Formula**:
```
L_layout = -1/B Σᵦ log p(ℓₜ₊₁⁽ᵇ⁾ | hₜ⁽ᵇ⁾; θₗ)
```

Where:
- ℓ_t+1 ∈ {EN, ZH, SYM, EMOJI, NUM} (5 layout classes)
- Cross-entropy over softmax probabilities

### 2.4 Language-Layout Consistency Loss (`L_consistency`)

**Purpose**: Ensure coherent multilingual behavior by penalizing mismatched predicted layout and expected language.

**Formula**:
```
L_consistency = -1/B Σᵦ log p(ℓ̂ₜ₊₁⁽ᵇ⁾ | target_lang⁽ᵇ⁾; θ)
```

Where:
- ℓ̂_t+1 = Predicted layout mode from layout_probs
- target_lang = Ground truth language label for the target text

**Implementation**: Simplified cross-entropy between predicted layout probabilities and target language labels, encouraging the layout predictor to be consistent with expected language context.

**Behavior**:
- Penalizes English text under Chinese layout mode (and vice versa)
- Maintains consistency between layout prediction and text generation

### 2.5 MoE Load Balancing Loss (`L_load`)

**Purpose**: Prevent expert collapse and ensure balanced utilization across language experts.

**Formula**:
```
L_load = Σᵢ (fᵢ - 1/5)²
```

Where:
- f_i = 1/B Σᵦ g_b,i (average routing probability to expert i)
- Target: 1/5 = 0.2 (uniform distribution across 5 experts)
- Experts: {EN, ZH, SYM, NUM, EMOJI}

**Implementation**: `KeyPilotLoss.load_balancing_loss()`

**Behavior**:
- Promotes robust multilingual capabilities
- Prevents over-reliance on specific experts
- Ensures all experts receive roughly equal utilization

## Loss Components Summary

| Component | Type | Weight | Formula | Purpose |
|-----------|------|--------|---------|---------|
| **Encoder Loss** | | λ_enc = 0.3 | | |
| ├─ Alignment | InfoNCE | λ_align = 1.0 | Contrastive | Vision-text alignment |
| └─ Contrastive | InfoNCE | λ_contrastive = 0.2 | Contrastive | ROI-text alignment |
| **Decoder Loss** | | λ_dec = 1.0 | | |
| ├─ Text Generation | CE | 1.0 | Token-level | Unified text tasks |
| ├─ Task Routing | CE | λ_task = 0.5 | Pseudo-labels | Dynamic task selection |
| ├─ Layout Prediction | CE | λ_layout = 0.4 | 5-class | Keyboard mode |
| ├─ Consistency | CE | λ_consistency = 0.3 | Lang-layout | Coherent behavior |
| └─ Load Balancing | Quadratic | λ_load = 0.01 | Expert usage | Prevent collapse |

## Implementation Details

### Code Location
- **Loss Class**: `src/keypilot/models/keypilot_model.py` → `KeyPilotLoss`
- **Training Script**: `scripts/train.py` → `train_epoch()`

### Usage in Training

```python
# 1. Create loss criterion
criterion = KeyPilotLoss(**config['training']['loss'])

# 2. Forward pass
outputs = model(image, input_ids, attention_mask, ...)

# 3. Prepare targets
targets = {
    'target_task': task_label,
    'target_layout': layout_label,
    'target_tokens': target_token_ids,
    'target_type': target_type,
}

# 4. Compute loss
loss_dict = criterion(outputs, targets)
loss_total = loss_dict['loss_total']

# 5. Backward pass
loss_total.backward()
```

### Loss Components in `loss_dict`

The `KeyPilotLoss.forward()` method returns a dictionary containing:

```python
{
    'loss_total': tensor(scalar),      # Total weighted loss
    'loss_encoder': tensor(scalar),    # Total encoder loss
    'loss_decoder': tensor(scalar),    # Total decoder loss
    'loss_align': tensor(scalar),      # Alignment loss
    'loss_contrastive': tensor(scalar),# Contrastive loss
    'loss_text': tensor(scalar),       # Text generation loss
    'loss_task': tensor(scalar),       # Task routing loss
    'loss_layout': tensor(scalar),     # Layout prediction loss
    'loss_consistency': tensor(scalar),# Consistency loss
    'loss_load': tensor(scalar),       # Load balancing loss
}
```

### Required Model Outputs

For the loss computation, the model's `forward()` method should return:

```python
outputs = {
    # Encoder outputs (for encoder losses)
    'global_feat': tensor([B, 256]),      # Global image feature
    'text_feat': tensor([B, 256]),        # Text CLS feature
    'roi_feats': tensor([B, 4, 256]),     # ROI features (optional)
    
    # Decoder outputs (for decoder losses)
    'logits': tensor([B, vocab_size]),    # Token prediction logits
    'task_probs': tensor([B, 3]),         # Task routing probabilities
    'layout_probs': tensor([B, 5]),       # Layout prediction logits
    'layout_id': tensor([B]),             # Predicted layout ID
    'expert_probs': tensor([B, 5]),       # MoE expert routing probs
    
    # Auxiliary outputs
    'h_t': tensor([B, 256]),              # Fused representation
}
```

## Hyperparameter Tuning Guide

### Temperature (τ)
- **Default**: 0.07
- **Range**: [0.05, 0.15]
- **Effect**: Lower values make contrastive learning more discriminative

### Encoder Weight (λ_enc)
- **Default**: 0.3
- **Range**: [0.1, 0.5]
- **Effect**: Balance between multimodal alignment and task performance

### Decoder Weight (λ_dec)
- **Default**: 1.0
- **Range**: [0.5, 2.0]
- **Effect**: Primary task optimization weight

### Task Loss Weight (λ_task)
- **Default**: 0.5
- **Range**: [0.3, 1.0]
- **Effect**: Importance of task routing accuracy

### Layout Loss Weight (λ_layout)
- **Default**: 0.4
- **Range**: [0.2, 0.8]
- **Effect**: Importance of layout prediction

### Consistency Weight (λ_consistency)
- **Default**: 0.3
- **Range**: [0.1, 0.5]
- **Effect**: Strength of language-layout alignment

### Load Balancing Weight (λ_load)
- **Default**: 0.01
- **Range**: [0.005, 0.05]
- **Effect**: Strength of expert load balancing (too high → reduced performance)

## Performance Impact

| Loss Component | Training Speed Impact | Model Performance Impact |
|----------------|----------------------|--------------------------|
| Alignment | Minimal (<5%) | High (multimodal understanding) |
| Contrastive | Minimal (<3%) | Medium (spatial awareness) |
| Text Generation | High (main task) | Critical (text quality) |
| Task Routing | Low (~5%) | High (task selection) |
| Layout Prediction | Low (~5%) | High (UX consistency) |
| Consistency | Minimal (<2%) | Medium (coherent behavior) |
| Load Balancing | Minimal (<1%) | Medium (multilingual robustness) |

## Debugging Tips

### High Loss Values

1. **loss_align > 5.0**: Poor vision-text alignment
   - Check image preprocessing
   - Verify text encoding
   - Reduce temperature τ

2. **loss_text > 10.0**: Poor text generation
   - Check vocabulary mapping
   - Verify target tokens
   - Adjust learning rate

3. **loss_load > 0.5**: Expert imbalance
   - Increase λ_load
   - Check routing logic
   - Verify expert initialization

### Loss Not Decreasing

1. Check data loading pipeline
2. Verify gradient flow (check for NaN/Inf)
3. Adjust learning rate schedule
4. Check for label mismatches

### Monitoring During Training

```python
# Log all loss components
logger.info(f"Epoch {epoch}:")
logger.info(f"  Total Loss: {loss_dict['loss_total']:.4f}")
logger.info(f"  Encoder Loss: {loss_dict['loss_encoder']:.4f}")
logger.info(f"    - Alignment: {loss_dict['loss_align']:.4f}")
logger.info(f"    - Contrastive: {loss_dict['loss_contrastive']:.4f}")
logger.info(f"  Decoder Loss: {loss_dict['loss_decoder']:.4f}")
logger.info(f"    - Text: {loss_dict['loss_text']:.4f}")
logger.info(f"    - Task: {loss_dict['loss_task']:.4f}")
logger.info(f"    - Layout: {loss_dict['loss_layout']:.4f}")
logger.info(f"    - Consistency: {loss_dict['loss_consistency']:.4f}")
logger.info(f"    - Load Balance: {loss_dict['loss_load']:.4f}")
```

## References

- Full technical details: `docs/method.md` (Section: Training Objective)
- Implementation: `src/keypilot/models/keypilot_model.py`
- Training script: `scripts/train.py`

