# KeyPilot Method

## Overview

In this section, we formally define the intelligent IME problem and provide a detailed technical solution for KeyPilot.

## System Architecture

![KeyPilot Architecture](images/architecture.png)

**Figure 1: The Architecture of KeyPilot.** The system encodes the screen image (\(I_t\)) and text context (\(C_t\)) via a lightweight vision–language encoder, and decodes the fused representation into task outputs—error detection, suggestion, auto-completion, and layout prediction (\(y_t, \ell_t\)).

The architecture consists of two main components:

1. **Vision-Language Encoder** (left): Processes multimodal inputs using MobileViT backbone, MobileSAM for ROI extraction, MobileCLIP for global features, and TinyLM for text encoding, producing a unified representation \(h_t\).

2. **Task-Specific Decoder** (right): Routes the encoded representation through task and layout routers to specialized language experts (Chinese, English, Emoji, Symbol) via a Language Router, generating task-specific outputs with vocabulary alignment.

## Vision-Language Encoder

We design a lightweight yet expressive multimodal encoder \(\mathbf{E}(\cdot)\) that jointly processes the *screen image* \(I_t \in \mathbb{R}^{H \times W \times 3}\), the *ongoing text sequence* \(C_t = \{w_1, \dots, w_t\}\), and an optional *user personality embedding* \(P_u \in \mathbb{R}^{64}\) to produce a unified contextual representation \(h_t \in \mathbb{R}^{256}\).

Unlike traditional vision-language models that rely on heavy backbones and GPU inference, our design prioritizes *on-device efficiency* through structural compression and token-level optimization. The entire encoder runs locally on mobile NPUs, quantized to INT8 precision, achieving an inference latency of under 19 ms while preserving strong semantic fidelity.

### Shared Visual Backbone

A MobileViT-XXS backbone (\(\alpha=0.75\)) serves as the shared visual feature extractor across both global and local streams. Given the resized input \(I_t \in \mathbb{R}^{512\times256\times3}\), the backbone outputs a dense feature map \(F \in \mathbb{R}^{192 \times H/4 \times W/4}\) that preserves spatial structure at 4× downsampling. Weight sharing across visual pathways minimizes parameter redundancy and facilitates knowledge transfer between global and region-aware features.

### Dual-Path Visual Encoding

To effectively model both *scene-level semantics* and *fine-grained UI elements*, we adopt a dual-path visual encoding architecture:

#### Global Semantic Stream

The global feature map \(F\) is average-pooled and passed through a MobileCLIP-Tiny projection head to yield a single global image token \([\text{IMG}] \in \mathbb{R}^{256}\). This branch encodes holistic scene semantics such as layout composition and conversational context.

#### Local ROI Stream

To localize task-relevant components (e.g., input field, chat bubble, keyboard region, and title bar), we employ a MobileSAM-inspired SAM-Lite segmentation module with four fixed, learnable prompts. It predicts four binary masks from \(F\), each representing a region of interest (ROI).

Each ROI is cropped from \(I_t\), letterbox-resized to \(112 \times 112\) (preserving aspect ratio), and re-encoded by the same MobileViT-XXS backbone (shared weights). The resulting features are projected to produce four localized visual tokens \([\text{ROI}_1], [\text{ROI}_2], [\text{ROI}_3], [\text{ROI}_4] \in \mathbb{R}^{256}\) that capture spatially distinct visual cues.

**Example ROI segmentation**: The system segments the screen into regions like input field, chat bubble, keyboard area, and title bar, allowing fine-grained understanding of UI context.

### Textual Context Encoding

The text history \(C_t\) (truncated to 64 tokens) is encoded using a distilled 2-layer RoBERTa-tiny model (hidden size 312, projected to 256). The \([\text{CLS}]\) embedding from the final layer serves as the textual context token \([\text{CLS}_{text}] \in \mathbb{R}^{256}\), representing the semantic intent of the current conversation or message input. This lightweight encoder maintains strong contextual alignment while reducing computation by 8× compared to full-sized transformers.

### User Personality Integration

Each user \(u\) is associated with a learned personality embedding \(P_u \in \mathbb{R}^{64}\), which is linearly projected to \(\mathbb{R}^{256}\) to form the token \([P]\). This embedding captures user-specific linguistic and behavioral patterns—such as tone preference, emoji usage frequency, and layout habits—allowing the model to adapt predictions to individual input styles.

### Cross-Modal Fusion

All tokens from the visual, textual, and personality branches are concatenated in a fixed order:

$$
\begin{aligned}
\text{[CLS]} 
&\;||\; [\text{IMG}] 
\;||\; [\text{ROI}_1] 
\;||\; [\text{ROI}_2] 
\;||\; [\text{ROI}_3] 
\;||\; [\text{ROI}_4] \\
&\;||\; [\text{SEP}] 
\;||\; [\text{CLS}_\text{text}] 
\;||\; \text{[P]}
\end{aligned}
$$

This sequence is processed by a single-layer transformer (*Cross-Former*) with 8 attention heads (\(d_{\text{model}}=256\), FFN dimension 1024), equipped with gated cross-attention and FlashAttention-2 for efficient token interaction. The output at the \([\text{CLS}]\) position is extracted as the final multimodal representation:

$$
h_t = \text{CrossFormer}(\text{sequence})[0] \in \mathbb{R}^{256}
$$

This representation fuses spatial, semantic, and personalized context in a compact form, serving as the shared embedding for all downstream decoders.

### Encoder Specifications

The complete encoder contains approximately **13.8M parameters** (about 5.8 MB after INT8 quantization) and achieves **≤ 19 ms latency** per frame on a mid-range smartphone NPU. The model's small footprint, parallelizable structure, and quantization-aware training make it practical for real-time deployment within mobile input methods.

## Task-Specific Decoders

The task-specific decoder translates the multimodal embedding \(h_t \in \mathbb{R}^{256}\) into textual and layout outputs through a lightweight, Mixture-of-Experts (MoE)–based architecture optimized for on-device deployment. It performs text generation (correction, auto-completion, suggestion) and layout prediction in parallel, maintaining consistency through cross-task synchronization.

### Task Routing

The task embedding \(e_{\text{task}} \in \mathbb{R}^{256}\) is dynamically selected from a compact, learnable task codebook:

$$
\mathcal{C}_{\text{task}} = \{e_{\texttt{<ERR>}}, e_{\texttt{<COMP>}}, e_{\texttt{<SUG>}}\} \in \mathbb{R}^{3 \times 256}
$$

A lightweight gating network—a 2-layer MLP with ReLU activation (256 → 128 → 3)—infers softmax probabilities directly from the multimodal latent \(h_t\):

$$
\begin{split}
&g_{\text{task}} = \mathrm{Softmax}\!\left(W_{\text{task}}^{(2)} \cdot \mathrm{ReLU}\!\left(W_{\text{task}}^{(1)} h_t + b_{\text{task}}^{(1)}\right) + b_{\text{task}}^{(2)}\right) \in \Delta^2 \\
&e_{\text{task}} = \sum_{i=1}^{3} g_{\text{task},i} \, \mathcal{C}_{\text{task}}[i]
\end{split}
$$

where \(W_{\text{task}}^{(1)} \in \mathbb{R}^{128 \times 256}\), \(W_{\text{task}}^{(2)} \in \mathbb{R}^{3 \times 128}\). For ambiguous contexts (e.g., partial input amenable to both correction and completion), we employ top-\(k\) activation (\(k=1\) or \(2\), triggered if \(\max g_{\text{task}} < 0.7\)) to enable weighted mixtures, enhancing robustness to code-switched or transitional keystrokes.

### Layout Routing

The layout hint embedding \(e_{\text{layout}} \in \mathbb{R}^{256}\) is selected from a learnable layout codebook:

$$
\mathcal{C}_{\text{layout}} = \{e_{\texttt{<EN>}}, e_{\texttt{<ZH>}}, e_{\texttt{<SYM>}}, e_{\texttt{<EMOJI>}}, e_{\texttt{<NUM>}}\} \in \mathbb{R}^{5 \times 256}
$$

via an independent layout router \(\mathcal{D}_l(\cdot)\). The router predicts the next keyboard mode \(\hat{\ell}_{t+1} \in \{\texttt{EN}, \texttt{ZH}, \texttt{SYM}, \texttt{EMOJI}, \texttt{NUM}\}\) directly from the multimodal representation \(h_t\).

Specifically, \(\mathcal{D}_l\) is a 2-layer causal transformer (hidden size 256, 8 attention heads, FFN dimension 1024) prepended with a learnable prefix token `<LAY>`:

$$
\begin{split}
z_l &= \mathrm{Concat}[e_{\texttt{<LAY>}}, h_t] \\
\tilde{z}_l &= \mathrm{Transformer}_{\text{lay}}(z_l; \Theta_l)
\end{split}
$$

where \(\Theta_l\) are the transformer parameters. The output corresponding to the prefix token is projected to logits over five layout classes:

$$
\begin{split}
\ell_{\text{logits}} &= W_l \tilde{z}_l[0] + b_l \\
\hat{\ell}_{t+1} &= \arg\max \mathrm{Softmax}(\ell_{\text{logits}})
\end{split}
$$

where \(\ell_{\text{logits}}\) are the raw score output for the five possible layouts.

#### Temporal Stability

To stabilize layout predictions across consecutive keystrokes, we incorporate a temporal bias from the prior layout \(\ell_t\):

$$
\ell_{\text{logits}} \leftarrow \ell_{\text{logits}} + \alpha \cdot \mathrm{onehot}(\ell_t), \quad \alpha = 0.3
$$

which acts as a soft hysteresis mechanism that discourages spurious mode switches. A layout switch is only accepted when the confidence of the new prediction exceeds a calibrated threshold:

$$
p_{\hat{\ell}_{t+1}} = 
\left[\mathrm{Softmax}(\ell_{\text{logits}})\right]_{\hat{\ell}_{t+1}} > 0.8
$$

Otherwise, the prior layout \(\ell_t\) is retained, ensuring temporal consistency and preventing flicker (reducing switch rate by 68% in ablation). Finally, the active layout embedding is retrieved as:

$$
e_{\text{layout}} = \mathcal{C}_{\text{layout}}[\hat{\ell}_{t+1}]
$$

which conditions the MoE gating network. For instance, selecting \(\hat{\ell}_{t+1} = \texttt{ZH}\) increases the routing logit of the Chinese expert by +0.4, effectively biasing decoding toward the corresponding language space.

### Input Representation

The input sequence is constructed as a concatenated embedding:

$$
x_t = \mathrm{Concat}\big[h_t, \, e_{\text{task}}, \, e_{\text{layout}}, \,  \underbrace{e_{\text{token}_1}, \dots, e_{\text{token}_{N-1}}}_{\text{autoregression}}\big]
$$

where \(x_t \in \mathbb{R}^{(3+N) \times 256}\).

For the *suggestion* task, the decoder is autoregressive with \(N-1\) previously predicted token embeddings \(\{e_{\text{token}_1}, \dots, e_{\text{token}_{N-1}}\}\) appended to the input to generate the next token. For the *error correction* and *auto-completion* tasks, \(N=0\) and \(N=1\), respectively, reflecting their single-step or short-context nature.

### Language MoE Decoder

This module serves as the core generative backbone of KeyPilot, unifying correction, completion, and suggestion within a sparse, language-specialized framework. A compact two-layer MLP router determines the activation weights over five language experts:

$$
\begin{split}
h_{\text{gate}} &= \sigma\!\left(W_1 x_t + b_1\right) \in \mathbb{R}^{128} \\
g &= \mathrm{Softmax}\!\left(W_2 h_{\text{gate}} + b_2\right) \in \Delta^4
\end{split}
$$

where \(x_t \in \mathbb{R}^{3 \times 256}\) is the flattened input embedding, \(W_1 \in \mathbb{R}^{128 \times 768}\) (from \(3\times256\) concatenation), \(W_2 \in \mathbb{R}^{5 \times 128}\), and \(\sigma(\cdot)\) denotes ReLU activation with dropout (\(p=0.1\)) for regularization.

This lightweight router (~0.1M parameters) integrates two conditioning signals to guide expert selection:

1. **Task token** \(e_{\text{task}}\): Biases routing toward appropriate experts (e.g., `<ERR>` increases activation of English and Symbol experts for typo-prone inputs)
2. **Layout hint** \(e_{\text{layout}}\): Injects language priors via a learned projection matrix \(W_{\text{hint}} \in \mathbb{R}^{5 \times 256}\)

Formally, the layout embedding produces an additive bias term \(\Delta_{\text{hint}} = W_{\text{hint}} e_{\text{layout}}\) that modifies the router logits before softmax:

$$
g = \mathrm{Softmax}\!\left(W_2 h_{\text{gate}} + b_2 + \Delta_{\text{hint}}\right)
$$

so that, for instance, a `<ZH>` layout increases the Chinese expert's activation by approximately +0.4 in logit space. This hybrid implicit–explicit conditioning mechanism effectively aligns language context with expert selection, achieving over 96% language–expert alignment on held-out data.

#### Expert Architecture

Five lightweight experts—**English (EN)**, **Chinese (ZH)**, **Symbol (SYM)**, **Number (NUM)**, and **Emoji (EMOJI)**—are implemented as one-layer causal Transformers (hidden size 256, 8 heads, FFN dimension 1024, GeLU activation), totaling ~2.4M parameters.

Each expert is initialized through language-biased distillation from a multilingual teacher model (e.g., Qwen3 or DeepSeek) and optimized using a combination of token-level cross-entropy and an auxiliary language classification objective:

$$
\mathcal{L}_{\text{lang}} = -\log q[\text{true\_lang}], \quad q = \mathrm{Softmax}(W_{\text{lang}} z_t)
$$

encouraging domain specialization (e.g., the ZH expert learns Hanzi co-occurrence patterns and pinyin-to-character mappings).

#### Sparse Aggregation and Decoding

The aggregated output representation is computed as:

$$
z_t = \sum_{i \in \mathcal{K}} g_i \cdot \mathcal{E}_i(x_t) \in \mathbb{R}^{256}
$$

where \(\mathcal{K}\) is the top-\(k\) expert index set and \(\mathcal{E}_i(\cdot)\) denotes each expert's Transformer block with residual and normalization layers. A top-1 policy is applied when \(\max(g) > 0.7\), otherwise top-2 with weighted mixing to preserve diversity.

For the *suggestion* task, \(z_t\) initializes autoregressive decoding with speculative sampling (lookahead 2, beam width 4) under causal masking, while *correction* and *completion* use non-autoregressive single-step projection.

The final representation \(z_t\) is projected onto a shared multilingual vocabulary (\(|V|=32\)K, mixed BPE covering English words, Chinese characters, symbols, and emojis):

$$
\begin{split}
\mathrm{logits}(y_t) &= W_o z_t + b_o \\ 
p(y_t) &= \mathrm{Softmax}\!\left(\mathrm{logits}(y_t)\right)
\end{split}
$$

where \(W_o \in \mathbb{R}^{32K \times 256}\). This unified output head enables smooth code-switching, while language-specific experts maintain intra-language fluency and style.

### MoE Architecture Benefits

The MoE architecture offers a principled balance between model capacity and computational efficiency:

- **High Specialization**: By activating only a small subset of language experts per input, KeyPilot achieves high specialization without increasing inference cost
- **Language-Specific Patterns**: Each expert captures language- or symbol-specific patterns, improving fluency and contextual accuracy
- **Dynamic Allocation**: The router dynamically allocates computation based on task and layout cues
- **Efficient Deployment**: This sparse, adaptive structure enables efficient on-device deployment by keeping active parameters minimal during each decoding step
- **Multilingual Support**: Enhanced multilingual generalization and code-switching capability

## Training Objective

The model is trained end-to-end using a multitask loss that jointly optimizes the vision–language encoder and task-specific decoders, while incorporating knowledge distillation for efficient transfer from larger pretrained models. This formulation preserves encoder–decoder modularity, allowing staged training and lightweight fine-tuning under mobile resource constraints:

$$
\mathcal{L}_{\text{total}} =
\lambda_{\text{enc}} \mathcal{L}_{\text{enc}} +
\lambda_{\text{dec}} \mathcal{L}_{\text{dec}} +
\lambda_{\text{dist}} \mathcal{L}_{\text{dist}}
$$

where \(\lambda_{\text{enc}}=0.3\), \(\lambda_{\text{dec}}=1.0\), and \(\lambda_{\text{dist}}=0.5\) are determined via grid search on a validation set of multimodal IME interactions. Decoupled encoder and decoder losses enable incremental updates (e.g., adapting to new layouts or user behaviors) without full retraining.

### Encoder Loss

The encoder \(\mathbf{E}(\cdot)\) is trained with self-supervised objectives that encourage robust cross-modal alignment among the screen image \(I_t\), text sequence \(C_t\), and user personality embedding \(P_u\). This ensures that the fused representation \(h_t\) captures semantically meaningful UI–text relationships crucial for proactive IME prediction:

$$
\mathcal{L}_{\text{enc}} =
\mathcal{L}_{\text{align}} +
0.2 \, \mathcal{L}_{\text{contrastive}}
$$

where both losses operate on intermediate tokens before the Cross-Former fusion layer.

#### Cross-Modal Alignment

We adopt a momentum contrastive learning framework to align global visual and textual embeddings without requiring paired supervision. For a batch of \(B\) samples, let \(v_i = [\text{IMG}]_i\) and \(t_i = [\text{CLS}_{\text{text}}]_i\) represent the visual and textual anchors from the same timestep. The alignment loss encourages matched pairs to be close in the shared space while pushing apart negatives:

$$
\mathcal{L}_{\text{align}} =
-\frac{1}{B} \sum_{i=1}^{B}
\log
\frac{
\exp(\text{sim}(v_i, t_i)/\tau)
}{
\sum_{j=1}^{B} \exp(\text{sim}(v_i, t_j)/\tau)
}
$$

where \(\text{sim}(\cdot,\cdot)\) denotes cosine similarity, \(\tau=0.07\) is the temperature, and a momentum-updated queue of negatives ensures scalable training. This objective allows the encoder to implicitly learn UI–text correspondences (e.g., associating chat bubbles with conversational intent).

#### Contrastive Regularization

To refine local feature alignment, we apply an auxiliary InfoNCE loss between region-of-interest (ROI) tokens and textual embeddings:

$$
\mathcal{L}_{\text{contrastive}} =
-\frac{1}{B} \sum_{i=1}^{B}
\log
\frac{
\exp(\text{sim}(\bar{r}_i, t_i)/\tau)
}{
\sum_{k=1}^{B} \exp(\text{sim}(\bar{r}_i, t_k)/\tau)
}
$$

where \(\bar{r}_i = \frac{1}{4}\sum_{m=1}^{4} [\text{ROI}_m]_i\) is the mean of four localized visual tokens. This fine-grained alignment promotes spatial awareness of interactive regions (e.g., focused input boxes or active cursors), improving robustness to UI layout variations.

### Decoder Loss

The decoder is optimized via a weighted multitask objective that unifies text generation and layout prediction, conditioned on \(h_t\) to ensure multimodal coherence:

$$
\mathcal{L}_{\text{dec}} =
\lambda_e \mathcal{L}_e +
\lambda_c \mathcal{L}_c +
\lambda_s \mathcal{L}_s +
\lambda_l \mathcal{L}_l +
\lambda_{\text{task}} \mathcal{L}_{\text{task}} +
\lambda_{\text{load}} \mathcal{L}_{\text{load}}
$$

where \(\mathcal{L}_e\), \(\mathcal{L}_c\), \(\mathcal{L}_s\), and \(\mathcal{L}_l\) correspond to error correction, auto-completion, suggestion, and layout prediction, respectively. Optimal weights \(\boldsymbol{\lambda} = [1.0, 0.8, 0.6, 0.4, 0.5, 0.01]\) are tuned to prioritize high-impact tasks like correction while maintaining MoE stability.

#### Text-Based Losses

All text tasks are trained with cross-entropy over a shared multilingual vocabulary (\(|V|=32\)K):

$$
\begin{align}
\mathcal{L}_e &= -\log p(\hat{w}_t \mid h_t, e_{\text{task}=\texttt{<ERR>}}; \theta) \\
\mathcal{L}_c &= -\log p(w_{t+1} \mid h_t, e_{\text{task}=\texttt{<COMP>}}, w_t; \theta) \\
\mathcal{L}_s &= -\sum_{n=1}^{N} \log p(s_{t+n} \mid h_t, e_{\text{task}=\texttt{<SUG>}}, s_{<n}; \theta)
\end{align}
$$

with \(N=5\) for the suggestion task and teacher forcing during training. Ground-truth targets are derived from user interaction logs, including post-edit corrections and accepted suggestions.

#### Layout Prediction Loss

The layout decoder is trained with cross-entropy combined with a language–layout consistency regularizer:

$$
\mathcal{L}_l = \mathcal{L}_{\text{CE}}(\hat{\ell}_{t+1} \mid h_t; \theta_l) + \beta \, \mathcal{L}_{\text{consistency}}
$$

where \(\beta=0.3\) and

$$
\mathcal{L}_{\text{consistency}} = -\log p(\text{Lang}(y_{t+1}) = \hat{\ell}_{t+1} \mid z_t; \theta)
$$

penalizes mismatched language–layout pairs (e.g., English text under `<ZH>` mode), promoting coherent multilingual switching.

#### Task Routing Supervision

To guide dynamic task selection, the gating distribution \(g_{\text{task}}\) is supervised using pseudo-labels derived from user actions (e.g., suggestion acceptance implies `<SUG>`):

$$
\mathcal{L}_{\text{task}} = -\sum_{i=1}^{3} y_i \log g_{\text{task},i}
$$

#### MoE Load Balancing

To avoid expert collapse in the language MoE, a quadratic load-balancing loss encourages uniform expert utilization:

$$
\mathcal{L}_{\text{load}} = \sum_{i=1}^{5} \left(f_i - \tfrac{1}{5}\right)^2, \quad f_i = \tfrac{1}{B}\sum_{b=1}^{B} g_{b,i}
$$

maintaining \(f_i \approx 0.2\) across the five experts (EN, ZH, SYM, NUM, EMOJI).

### Multitask Decoder Loss Components

| Task | Loss Type | Target | Weight | Notes |
|------|-----------|--------|---------|-------|
| Error Correction | CE | Corrected token \(\hat{w}_t\) | 1.0 | Single-step |
| Auto-Completion | CE | Next token \(w_{t+1}\) | 0.8 | Short AR |
| Suggestion | CE | Sequence \(s_{t+1:N}\) | 0.6 | AR (\(N=5\)) |
| Layout Prediction | CE + Consistency | Mode \(\ell_{t+1}\) | 0.4 | Hysteresis-aware |
| Task Routing | CE | Pseudo-label | 0.5 | From logs |
| MoE Balancing | Quadratic | Uniform \(f_i\) | 0.01 | Anti-collapse |

### Distillation Loss

To enhance efficiency and generalization, we incorporate knowledge distillation from a pretrained multilingual vision–language teacher (e.g., DeepSeek OCR):

$$
\mathcal{L}_{\text{dist}} = \mathcal{L}_{\text{KL}} + 0.5 \, \mathcal{L}_{\text{feat}}
$$

where \(\mathcal{L}_{\text{KL}}\) minimizes the divergence between student and teacher token logits, and \(\mathcal{L}_{\text{feat}}\) aligns intermediate encoder representations (\(h_t\)) via mean squared error. This distillation transfers rich multimodal priors, reducing cold-start errors and accelerating adaptation in low-data mobile environments.

## Summary

Overall, this composite objective balances multimodal alignment, multitask supervision, and efficient specialization, enabling KeyPilot to achieve:

- **High Accuracy**: Through multimodal understanding and task-specific optimization
- **Low Latency**: Via efficient encoder design and sparse MoE decoding
- **Adaptive Layout Control**: Through joint language-layout consistency
- **Real-world Deployment**: Practical for mobile typing scenarios with on-device constraints

The architecture achieves:
- **Encoder**: 13.8M parameters, 5.8 MB (INT8), ≤19 ms latency
- **Decoder**: ~2.4M parameters (MoE experts), sparse activation
- **Total**: Highly efficient for real-time mobile deployment

---

*This document details the technical method for KeyPilot. For implementation, see the source code in `src/keypilot/models/`.*

