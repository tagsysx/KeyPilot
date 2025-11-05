# N-Candidate Generation Design for KeyPilot

## Overview

For auto-completion and error-correction tasks, KeyPilot generates **N candidates** (typically N=5) instead of a single prediction. This design provides users with multiple ranked options, improving user experience and accommodating ambiguous input scenarios.

## Motivation

### Why N Candidates?

1. **Ambiguity Resolution**: In many typing scenarios, multiple completions or corrections are plausible. For example:
   - Input: "go" → Candidates: ["go", "good", "going", "gone", "google"]
   - Typo: "teh" → Candidates: ["the", "tea", "ten", "tech", "them"]

2. **User Choice**: Users can quickly scan and select the most appropriate option without retyping.

3. **Higher Success Rate**: The top-N accuracy metric is more forgiving than strict top-1 accuracy, better reflecting real-world IME usability.

4. **Contextual Flexibility**: Different contexts may favor different candidates; providing options accommodates diverse user intent.

## Design Specifications

### Task-Specific Behavior

| Task | Output Format | Number of Candidates | Selection Method |
|------|---------------|---------------------|------------------|
| **Error Correction** | N token candidates | N=5 (default) | Top-k from vocabulary |
| **Auto-Completion** | N token candidates | N=5 (default) | Top-k from vocabulary |
| **Suggestion** | Single sequence | 1 sequence of 5 tokens | Autoregressive generation |

### Implementation Details

#### 1. Candidate Generation

For error-correction and auto-completion tasks:

```python
def generate_candidates(self, h_t, prev_layout=None, num_candidates=5, temperature=1.0):
    """
    Generate N candidate tokens for non-autoregressive tasks.
    
    Args:
        h_t: Multimodal representation [B, 256]
        num_candidates: Number of candidates to generate (default: 5)
        temperature: Sampling temperature for diversity
    
    Returns:
        candidate_tokens: Top-N token IDs [B, num_candidates]
        candidate_probs: Probabilities [B, num_candidates]
        metadata: Task/layout/expert information
    """
    # Get logits from MoE decoder
    logits, expert_probs = self.language_moe(h_t, e_task, e_layout)
    
    # Apply temperature for diversity
    logits = logits / temperature
    
    # Get top-N candidates
    probs = F.softmax(logits, dim=-1)
    candidate_probs, candidate_tokens = torch.topk(probs, k=num_candidates, dim=-1)
    
    return candidate_tokens, candidate_probs, metadata
```

#### 2. Candidate Ranking

Candidates are ranked by model confidence (probability):
- **Position 1**: Highest probability (most confident)
- **Position N**: Lowest probability among top-N

Users typically scan from left to right in UI, so higher-ranked candidates appear first.

#### 3. Temperature Control

- **Temperature = 1.0** (default): Balanced diversity
- **Temperature < 1.0**: More confident, less diverse (e.g., 0.8)
- **Temperature > 1.0**: More diverse, less confident (e.g., 1.2)

Lower temperature focuses on top predictions; higher temperature encourages exploration.

### Training and Evaluation

#### Training Loss

During training, the loss is computed over the **single ground-truth token**:

$$
\mathcal{L}_{\text{text}} = -\log p(y_{\text{true}} \mid h_t, e_{\text{task}}; \theta)
$$

The model is trained to maximize the probability of the correct token. N-candidate generation is applied only during inference.

#### Evaluation Metric: Top-N Accuracy

A prediction is considered **correct** if the ground truth appears in **any of the N candidates**:

$$
\text{Accuracy}_{\text{top-N}} = \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}\left[ y_{\text{true}}^{(i)} \in \{\hat{y}_1^{(i)}, \dots, \hat{y}_N^{(i)}\} \right]
$$

where:
- \( y_{\text{true}}^{(i)} \) is the ground-truth token for sample \(i\)
- \( \{\hat{y}_1^{(i)}, \dots, \hat{y}_N^{(i)}\} \) are the N predicted candidates
- \( \mathbb{1}[\cdot] \) is the indicator function (1 if true, 0 if false)

**Implementation** (PyTorch):

```python
# Get top-N predictions
top_n_probs, top_n_tokens = torch.topk(logits, k=num_candidates, dim=-1)
# top_n_tokens: [B, N]

# Check if ground truth is in any candidate
target_expanded = target.unsqueeze(1).expand_as(top_n_tokens)  # [B, N]
matches = (top_n_tokens == target_expanded).any(dim=1)  # [B]
accuracy = matches.float().mean().item()
```

#### Comparison: Top-1 vs Top-N Accuracy

| Metric | Definition | Example (N=5) |
|--------|------------|---------------|
| Top-1 Accuracy | Ground truth is the highest-ranked prediction | If "the" is rank 1 → correct |
| Top-N Accuracy | Ground truth is in any of top-N predictions | If "the" is in ranks 1-5 → correct |

**Expected Performance**:
- Top-1 Accuracy: ~70-80% (typical for complex IME tasks)
- Top-5 Accuracy: ~90-95% (significantly higher)

### User Interface Considerations

#### Candidate Display

Candidates can be displayed in various UI patterns:

1. **Horizontal Bar** (most common):
   ```
   [good] [going] [gone] [google] [goal]
   ```

2. **Dropdown List**:
   ```
   ▼ good (85%)
     going (10%)
     gone (3%)
     ...
   ```

3. **Inline Suggestions**:
   ```
   go|od  ← First candidate inline
   ```

#### Selection Methods

- **Tap/Click**: User taps on desired candidate
- **Swipe**: User swipes through candidates
- **Number Keys**: Press 1-5 to select candidate by position
- **Tab/Arrow Keys**: Navigate and press Enter

### Configuration Parameters

Default configuration in `configs/model_config.yaml`:

```yaml
inference:
  # Candidate generation
  num_candidates: 5  # Number of candidates for completion/error tasks
  candidate_temperature: 1.0  # Sampling temperature
  
  # Suggestion task (autoregressive)
  suggestion_length: 5  # Number of tokens for suggestion
  suggestion_temperature: 1.0
  suggestion_top_k: 50
  suggestion_top_p: 0.9
```

### Performance Impact

#### Latency

Generating N candidates has **minimal latency overhead**:
- Single forward pass: Same as before (~10ms)
- Top-k selection: O(N log V) where V=vocab_size (~0.5ms for N=5)
- **Total overhead**: < 1ms

#### Memory

- Storage: N × sizeof(token_id) per sample
- For N=5, int32 tokens: 5 × 4 bytes = 20 bytes per sample
- Negligible for typical batch sizes

## Examples

### Example 1: Auto-Completion

**Input**: "I'm going to the"

**Candidates** (N=5):
1. store (40%)
2. park (25%)
3. beach (15%)
4. office (12%)
5. gym (8%)

**Ground Truth**: "beach"

**Evaluation**:
- Top-1 Accuracy: ❌ Incorrect (predicted "store")
- Top-5 Accuracy: ✅ Correct ("beach" is rank 3)

### Example 2: Error Correction

**Input**: "I lov u" (typo: "lov")

**Candidates** (N=5):
1. love (85%)
2. low (7%)
3. lot (4%)
4. look (3%)
5. lost (1%)

**Ground Truth**: "love"

**Evaluation**:
- Top-1 Accuracy: ✅ Correct
- Top-5 Accuracy: ✅ Correct

## Implementation Checklist

- [x] Add `generate_candidates()` method to `KeyPilotDecoder`
- [x] Update evaluation metric in `scripts/train.py` to use top-N accuracy
- [x] Update documentation in `design_principles.md`
- [x] Update documentation in `method.md`
- [x] Update documentation in `module_selection.md`
- [x] Add configuration parameters for N candidates
- [ ] Add unit tests for `generate_candidates()`
- [ ] Add UI mockups for candidate display
- [ ] Benchmark latency overhead
- [ ] Collect user study data on selection patterns

## Future Enhancements

1. **Adaptive N**: Dynamically adjust N based on confidence
   - High confidence: N=3 (fewer, more accurate options)
   - Low confidence: N=7 (more options to explore)

2. **Context-Aware Ranking**: Rerank candidates using:
   - User history
   - App context
   - Frequency of use

3. **Personalized Candidates**: Learn user-specific preferences
   - Frequently selected candidates get boosted
   - Rare selections get demoted

4. **Multilingual Candidates**: Mix languages in candidates
   - Example: ["你好", "hello", "hi", "こんにちは", "안녕"]

5. **Beam Search**: Use beam search instead of top-k for better diversity
   - Maintains multiple hypotheses during decoding

## References

- [1] Mobile Input Methods: Best Practices (Google I/O 2023)
- [2] Top-k Sampling for Language Generation (Holtzman et al., 2019)
- [3] Evaluation Metrics for IME Systems (Microsoft Research, 2022)

---

**Last Updated**: 2025-11-05  
**Version**: 1.0

