# KeyPilot Design Principles

## Overview

In this section, we formally define the intelligent IME problem and provide a high-level solution.

## Design Principles

### Core Functions

KeyPilot extends conventional IMEs by integrating on-screen perception with joint vision-language context understanding. It provides four core functionalities:

- **Auto-Completion**: Predicts the next word or phrase based on the user's partial input and surrounding visual-textual context, enabling content-aware completion beyond standard language models. Provides **N candidate suggestions** (typically N=5) ranked by confidence, allowing users to select the most appropriate completion.

- **Semantic Error Correction**: Detects and corrects typos or inconsistencies by jointly analyzing visual and textual context (e.g., "I ate appl" near an apple image → "I ate an apple"). Multimodal cues improve correction accuracy over language-only approaches. Offers **N candidate corrections** (typically N=5) for user selection, accommodating ambiguous cases.

- **Adaptive Layout Switching**: Dynamically selects the optimal keyboard layout (e.g., QWERTY for English, T9 for Chinese, numeric, emoji, handwriting) based on input field semantics, app context, and visual scene, reducing manual layout switching.

- **Proactive Suggestion**: Anticipates user intent and offers relevant phrases before typing (e.g., opening an email draft triggers "Reply: Thanks, see you tomorrow"). This transforms the IME into a predictive assistant.

In practice, our model can be extended to support additional advanced functionalities such as *Form Auto-Fill*, *Accessibility Enhancement*, and *Screenshot-to-Text*. In this work, however, we focus on the four core functions.

### Challenges

We aim to satisfy two widely recognized criteria for IME systems:

- **Real-time Responsiveness**: The system must deliver predictions and interface updates with minimal latency (e.g., < 50 ms) to ensure a seamless typing experience.

- **On-device Efficiency**: The model should operate effectively within mobile hardware constraints, maintaining low memory usage (e.g., < 50 M) and computational cost without relying on cloud resources.

These design goals ensure that KeyPilot achieves a balanced trade-off between intelligence and efficiency, making it both practical for real-world deployment and scalable across diverse mobile platforms.

## Workflow

The overall workflow of KeyPilot operates as follows:

When the IME is activated (e.g., by focus or keypress), the system captures the current screen image \(I_t\) and the contextual text history \(C_t\) before the IME is closed. These inputs are jointly encoded by a vision-language (VL) model in a single forward pass, yielding a multimodal representation \(h_t\).

The representation \(h_t\) is then routed to four task-specific decoders corresponding to the functionalities introduced earlier:
- Error detection: \(\mathcal{D}_e(h_t)\)
- Auto-completion: \(\mathcal{D}_a(h_t)\)
- Layout adaptation: \(\mathcal{D}_l(h_t)\)
- Suggestion: \(\mathcal{D}_s(h_t)\)

### Keyboard Design

To ensure a consistent user experience, we adopt a traditional virtual keyboard design. The interface consists of two main areas:

- **Upper Area**: Displays textual outputs (corrections, suggestions, or completions)
- **Lower Area**: Presents the keyboard layout

Users can either select a suggested phrase directly or input new text using the layout.

### Output Priority

The displayed output follows a fixed priority order:

**Correction → Suggestion → Auto-Completion**

Correction results always take precedence; when no correction is available, the system turns to suggestion or auto-completion. The layout decoder predicts the most appropriate keyboard configuration (e.g., Chinese, English, or emoji layouts), which is immediately rendered in the lower area for user interaction. This process is repeated continuously as each new word is entered.

## Problem Formulation

We formulate the intelligent IME as a *multimodal perception-and-generation* problem that maps the user's current on-screen and textual context to adaptive keyboard outputs.

At each interaction step \(t\), the system observes:
- **Screen image** \(I_t\): Capturing the UI and conversation context
- **Textual sequence** \(C_t = \{w_1, w_2, \dots, w_t\}\): Representing the ongoing input history

The objective is to generate:
- **Contextually appropriate response** \(y_t\): Such as a suggestion, correction, or completion
- **Layout configuration** \(\ell_t\): Determining the next active keyboard mode (e.g., English, Chinese, emoji, or numeric)

### Encoding

The vision–language encoder jointly embeds visual, textual, and personalized contextual cues into a unified representation:

\[
h_t = \mathbf{E}(I_t, C_t, P_u)
\]

where \(P_u\) denotes an optional *personality embedding* that captures user \(u\)'s typing style, tone preference, or habitual layout usage. This embedding enables the model to adapt its predictions to user-specific behaviors. The encoder output \(h_t\) integrates semantic, visual, and personalized context, forming a compact multimodal representation for downstream decoders.

### Decoding

The encoded representation \(h_t\) is processed by four task-specific decoders, each responsible for a distinct IME function:

- **Error Detection**: \(\mathcal{D}_e(h_t) \to \hat{w}_t\)
- **Auto-Completion**: \(\mathcal{D}_a(h_t) \to w_{t+1}\)
- **Suggestion**: \(\mathcal{D}_s(h_t) \to s_{t+1}\)
- **Layout Adaptation**: \(\mathcal{D}_l(h_t) \to \ell_{t+1}\)

Each decoder operates independently but shares the same latent representation.

### Architecture

![KeyPilot Architecture](images/architecture.png)

**Figure: The Architecture of KeyPilot.** The complete system showing the vision-language encoder (left) and task-specific decoder (right).

The system architecture follows this flow:

1. **Input**: Screen image (\(I_t\)) and text context (\(C_t\))
2. **Encoder**: Lightweight vision–language encoder fuses multimodal inputs
   - MobileViT for visual backbone
   - MobileSAM for ROI extraction
   - MobileCLIP for global features
   - TinyLM for text encoding
3. **Decoders**: Four task-specific decoders process the fused representation
   - Task Router selects between error detection, completion, and suggestion
   - Layout Router predicts optimal keyboard layout
   - Language Router dispatches to specialized experts (EN, ZH, Emoji, Symbol)
4. **Output**: Task outputs—error detection, suggestion, auto-completion, and layout prediction (\(y_t, \ell_t\))

### Output Consistency

At each step, the system jointly determines the textual output \(y_t\) and the corresponding keyboard layout \(\ell_t\).

The textual output follows a fixed priority order:

\[
y_t =
\begin{cases}
s_{t+1:N}, & \text{if } C_t = \varnothing \quad \text{(Suggestion)} \\
\hat{w}_t, & \text{if } w_t \text{ is wrongly spelled} \\
w_{t+1}, & \text{otherwise (Auto-Completion)}
\end{cases}
\]

While the layout decoder \(\mathcal{D}_l\) continuously predicts \(\ell_t\), the generated text must remain consistent with the selected layout.

Formally, we define:

\[
(y_t, \ell_t) = \arg\max_{(y, \ell)} p(y, \ell \mid h_t), \quad \text{s.t. } \text{Lang}(y) = \text{Lang}(\ell)
\]

where:
- \(\text{Lang}(y)\) denotes the language inferred from the generated text (e.g., English, Chinese, emoji, or numeric)
- \(\text{Lang}(\ell)\) represents the language type of the predicted keyboard layout

**Example**: If \(y_t\) is a Chinese phrase, \(\ell_t\) must correspond to the Chinese layout, whereas an emoji output activates the emoji keyboard.

This joint consistency mechanism ensures that language generation and interface adaptation remain synchronized, providing seamless and context-aware user interaction.

## Key Design Decisions

### 1. Multimodal Context Understanding

The system jointly processes visual and textual information rather than treating them independently. This enables:
- Context-aware predictions based on what's displayed on screen
- Better understanding of user intent from UI elements
- Improved accuracy in ambiguous situations

### 2. Task-Specific Decoders

Rather than a single monolithic model, we use specialized decoders for each task:
- **Modularity**: Each decoder can be optimized independently
- **Efficiency**: Only relevant decoders need to be active
- **Maintainability**: Easier to update or improve individual components

### 3. Personality Embeddings

User-specific adaptations through \(P_u\) enable:
- Personalized predictions matching user's writing style
- Adaptation to frequently used phrases or expressions
- Learning user's keyboard layout preferences

### 4. Priority-Based Output

The fixed priority (Correction → Suggestion → Auto-Completion) ensures:
- Critical corrections are never missed
- Proactive assistance when no errors exist
- Fallback to basic auto-completion
- Consistent and predictable user experience

### 5. Language-Layout Consistency

Enforcing \(\text{Lang}(y) = \text{Lang}(\ell)\) guarantees:
- No mismatch between predicted text and available keyboard
- Seamless transitions between languages
- Reduced user frustration from layout mismatches

## Performance Requirements

### Latency Targets

| Operation | Target Latency |
|-----------|----------------|
| Encoder forward pass | < 30 ms |
| Decoder inference | < 10 ms |
| Layout switching | < 5 ms |
| Total end-to-end | < 50 ms |

### Resource Constraints

| Resource | Constraint |
|----------|------------|
| Model size | < 50 MB |
| RAM usage | < 100 MB |
| CPU utilization | < 30% average |
| Battery impact | Minimal (< 5% daily) |

## Future Extensions

While the current implementation focuses on four core functions, the architecture can be extended to support:

1. **Form Auto-Fill**: Intelligent completion of form fields based on screen context
2. **Accessibility Enhancement**: Voice-to-text with context awareness
3. **Screenshot-to-Text**: Extract and edit text from images
4. **Multi-language Mixing**: Handle code-switching within single conversations
5. **Gesture Recognition**: Swipe-based input with context prediction
6. **Emoji Recommendation**: Context-aware emoji suggestions

---

*This document outlines the design principles for KeyPilot, an intelligent vision-language typing agent. For implementation details, see the source code in `src/keypilot/`.*

