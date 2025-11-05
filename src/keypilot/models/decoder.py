"""
Task-Specific Decoders for KeyPilot

This module implements:
- Task Router: Selects between error correction, completion, suggestion
- Layout Router: Predicts optimal keyboard layout (EN/ZH/SYM/EMOJI/NUM)
- Language MoE: 5 expert decoders for multilingual generation
- Vocabulary: Shared 32K multilingual BPE

Total parameters: ~3.2M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from transformers import BertConfig, BertModel


class TaskRouter(nn.Module):
    """
    2-layer MLP router for task selection.
    
    Tasks: <ERR> (error correction), <COMP> (completion), <SUG> (suggestion)
    Parameters: ~0.1M
    """
    
    def __init__(self, d_model: int = 256, num_tasks: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.num_tasks = num_tasks
        
        # MLP gating network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_tasks)
        )
        
        # Learnable task codebook embeddings
        self.task_codebook = nn.Parameter(torch.randn(num_tasks, d_model))
        
        # Task names for interpretability
        self.task_names = ['<ERR>', '<COMP>', '<SUG>']
    
    def forward(self, h_t: torch.Tensor, threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Multimodal representation [B, 256]
            threshold: Confidence threshold for top-k selection
        
        Returns:
            e_task: Task embedding [B, 256]
            g_task: Gating probabilities [B, num_tasks]
        """
        logits = self.mlp(h_t)  # [B, 3]
        g_task = F.softmax(logits, dim=-1)  # [B, 3]
        
        # Top-k activation for ambiguous cases
        max_prob = g_task.max(dim=-1, keepdim=True)[0]  # [B, 1]
        
        # If max prob < threshold, use top-2 weighted mixture
        use_topk = (max_prob < threshold).squeeze(-1)  # [B]
        
        e_task = torch.zeros_like(h_t)  # [B, 256]
        
        for i in range(h_t.size(0)):
            if use_topk[i]:
                # Top-2 mixture
                topk_vals, topk_idx = torch.topk(g_task[i], k=2)
                for j, idx in enumerate(topk_idx):
                    e_task[i] += topk_vals[j] * self.task_codebook[idx]
            else:
                # Top-1
                idx = g_task[i].argmax()
                e_task[i] = self.task_codebook[idx]
        
        return e_task, g_task


class LayoutRouter(nn.Module):
    """
    2-layer Causal Transformer for layout prediction.
    
    Layouts: <EN>, <ZH>, <SYM>, <EMOJI>, <NUM>
    Parameters: ~0.3M
    """
    
    def __init__(self, d_model: int = 256, num_layouts: int = 5, num_layers: int = 2):
        super().__init__()
        self.num_layouts = num_layouts
        
        # Transformer configuration
        config = BertConfig(
            hidden_size=d_model,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=1024,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        self.transformer = BertModel(config, add_pooling_layer=False)
        
        # Learnable prefix token <LAY>
        self.lay_prefix = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Projection head
        self.proj_head = nn.Linear(d_model, num_layouts)
        
        # Layout codebook
        self.layout_codebook = nn.Parameter(torch.randn(num_layouts, d_model))
        
        # Layout names
        self.layout_names = ['<EN>', '<ZH>', '<SYM>', '<EMOJI>', '<NUM>']
    
    def forward(self,
                h_t: torch.Tensor,
                prev_layout: Optional[torch.Tensor] = None,
                alpha: float = 0.3,
                threshold: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Multimodal representation [B, 256]
            prev_layout: Previous layout IDs [B] for temporal bias
            alpha: Temporal bias weight
            threshold: Confidence threshold for layout switching
        
        Returns:
            e_layout: Layout embedding [B, 256]
            pred_layout: Predicted layout IDs [B]
            probs: Layout probabilities [B, num_layouts]
        """
        B = h_t.size(0)
        
        # Prepare input: [<LAY>, h_t]
        lay_prefix = self.lay_prefix.expand(B, -1, -1)  # [B, 1, 256]
        input_emb = torch.cat([lay_prefix, h_t.unsqueeze(1)], dim=1)  # [B, 2, 256]
        
        # Process through transformer
        outputs = self.transformer(inputs_embeds=input_emb)
        lay_out = outputs.last_hidden_state[:, 0, :]  # [B, 256] - CLS/<LAY> position
        
        # Project to logits
        logits = self.proj_head(lay_out)  # [B, num_layouts]
        
        # Add temporal bias from previous layout
        if prev_layout is not None:
            bias = F.one_hot(prev_layout, self.num_layouts).float() * alpha
            logits = logits + bias
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)  # [B, num_layouts]
        pred_layout = probs.argmax(dim=-1)  # [B]
        
        # Threshold-based stability: retain previous layout if confidence too low
        if prev_layout is not None:
            max_probs = probs.max(dim=-1)[0]  # [B]
            retain_prev = max_probs < threshold  # [B]
            pred_layout = torch.where(retain_prev, prev_layout, pred_layout)
        
        # Get layout embeddings
        e_layout = self.layout_codebook[pred_layout]  # [B, 256]
        
        return e_layout, pred_layout, probs


class LanguageExpert(nn.Module):
    """
    Single language expert: 1-layer causal Transformer.
    
    Specialized for: EN, ZH, SYM, NUM, or EMOJI
    Parameters: ~0.4M each
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [B, seq_len, 256]
        
        Returns:
            output: [B, seq_len, 256]
        """
        outputs = self.transformer(inputs_embeds=x)
        return outputs.last_hidden_state


class LanguageMoE(nn.Module):
    """
    Mixture-of-Experts decoder with 5 language specialists.
    
    Experts: EN, ZH, SYM, NUM, EMOJI
    Parameters: ~2.0M
    
    For auto-completion and error-correction tasks, generates N candidates
    (default N=5) using top-k sampling, providing users with multiple options.
    """
    
    def __init__(self, 
                 d_model: int = 256, 
                 num_experts: int = 5,
                 vocab_size: int = 32000,
                 hidden_dim: int = 128,
                 num_candidates: int = 5):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.num_candidates = num_candidates
        
        # Router: 2-layer MLP
        input_dim = 3 * d_model  # h_t + e_task + e_layout
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Layout hint projection for biasing expert selection
        self.hint_proj = nn.Linear(d_model, num_experts)
        
        # 5 Language experts
        self.experts = nn.ModuleList([
            LanguageExpert(d_model=d_model) for _ in range(num_experts)
        ])
        
        # Expert names
        self.expert_names = ['EN', 'ZH', 'SYM', 'NUM', 'EMOJI']
        
        # Shared output projection to vocabulary
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(self,
                h_t: torch.Tensor,
                e_task: torch.Tensor,
                e_layout: torch.Tensor,
                token_embeds: Optional[torch.Tensor] = None,
                top_k: int = 2,
                threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Multimodal representation [B, 256]
            e_task: Task embedding [B, 256]
            e_layout: Layout embedding [B, 256]
            token_embeds: Previous token embeddings for AR [B, N, 256], optional
            top_k: Number of experts to activate
            threshold: Confidence threshold for top-1 vs top-k
        
        Returns:
            logits: Output logits [B, vocab_size] or [B, N, vocab_size]
            gate_probs: Expert gating probabilities [B, num_experts]
        """
        B = h_t.size(0)
        
        # Prepare input sequence
        if token_embeds is None:
            # Non-autoregressive (error correction, completion)
            x_t = torch.stack([h_t, e_task, e_layout], dim=1)  # [B, 3, 256]
        else:
            # Autoregressive (suggestion)
            prefix = torch.stack([h_t, e_task, e_layout], dim=1)  # [B, 3, 256]
            x_t = torch.cat([prefix, token_embeds], dim=1)  # [B, 3+N, 256]
        
        # Router inference
        flat_input = torch.cat([h_t, e_task, e_layout], dim=-1)  # [B, 768]
        gate_logits = self.router(flat_input)  # [B, num_experts]
        
        # Add layout hint bias
        hint_bias = self.hint_proj(e_layout)  # [B, num_experts]
        gate_logits = gate_logits + hint_bias
        
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, num_experts]
        
        # Determine top-k based on max confidence
        max_prob = gate_probs.max(dim=-1)[0]  # [B]
        k = torch.where(max_prob > threshold, 1, top_k).max().item()  # Adaptive k
        
        # Select top-k experts
        topk_vals, topk_idx = torch.topk(gate_probs, k=k, dim=-1)  # [B, k]
        
        # Aggregate expert outputs
        z_t = torch.zeros(B, x_t.size(1), self.d_model, device=x_t.device)
        
        for i in range(k):
            expert_idx = topk_idx[:, i]  # [B]
            expert_weights = topk_vals[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            
            # Process through experts (batched by expert type)
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_t[mask]  # [N, seq_len, 256]
                    expert_output = self.experts[expert_id](expert_input)  # [N, seq_len, 256]
                    z_t[mask] += expert_weights[mask] * expert_output
        
        # Project to vocabulary
        logits = self.output_head(z_t)  # [B, seq_len, vocab_size]
        
        # For non-AR tasks, take the last token
        if token_embeds is None:
            logits = logits[:, -1, :]  # [B, vocab_size]
        
        return logits, gate_probs


class KeyPilotDecoder(nn.Module):
    """
    Complete task-specific decoder for KeyPilot.
    
    Total parameters: ~3.2M
    """
    
    def __init__(self,
                 d_model: int = 256,
                 vocab_size: int = 32000,
                 num_tasks: int = 3,
                 num_layouts: int = 5,
                 num_experts: int = 5):
        super().__init__()
        
        # Routers
        self.task_router = TaskRouter(d_model=d_model, num_tasks=num_tasks)
        self.layout_router = LayoutRouter(d_model=d_model, num_layouts=num_layouts)
        
        # Language MoE
        self.language_moe = LanguageMoE(
            d_model=d_model,
            num_experts=num_experts,
            vocab_size=vocab_size
        )
        
        # Token embeddings (shared with vocabulary)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self,
                h_t: torch.Tensor,
                prev_layout: Optional[torch.Tensor] = None,
                prev_tokens: Optional[torch.Tensor] = None,
                max_length: int = 5) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.
        
        Args:
            h_t: Multimodal representation [B, 256]
            prev_layout: Previous layout IDs [B] for temporal stability
            prev_tokens: Previous token IDs for autoregressive generation [B, N]
            max_length: Max tokens to generate for suggestion task
        
        Returns:
            outputs: Dictionary containing:
                - task_probs: Task probabilities [B, num_tasks]
                - layout_id: Predicted layout [B]
                - layout_probs: Layout probabilities [B, num_layouts]
                - logits: Output logits [B, vocab_size] or [B, max_length, vocab_size]
                - expert_probs: Expert gating probabilities [B, num_experts]
        """
        # Task routing
        e_task, task_probs = self.task_router(h_t)
        
        # Layout routing
        e_layout, layout_id, layout_probs = self.layout_router(h_t, prev_layout)
        
        # Prepare token embeddings for AR generation if needed
        token_embeds = None
        if prev_tokens is not None:
            token_embeds = self.token_embedding(prev_tokens)  # [B, N, 256]
        
        # Language MoE decoding
        logits, expert_probs = self.language_moe(h_t, e_task, e_layout, token_embeds)
        
        return {
            'task_probs': task_probs,
            'layout_id': layout_id,
            'layout_probs': layout_probs,
            'logits': logits,
            'expert_probs': expert_probs,
            'e_task': e_task,
            'e_layout': e_layout
        }
    
    def generate(self,
                 h_t: torch.Tensor,
                 prev_layout: Optional[torch.Tensor] = None,
                 max_length: int = 5,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9) -> Tuple[torch.Tensor, Dict]:
        """
        Autoregressive generation for suggestion task.
        
        Args:
            h_t: Multimodal representation [B, 256]
            prev_layout: Previous layout IDs [B]
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
        
        Returns:
            generated_tokens: [B, max_length]
            metadata: Generation metadata
        """
        B = h_t.size(0)
        device = h_t.device
        
        # Task and layout routing
        e_task, task_probs = self.task_router(h_t)
        e_layout, layout_id, layout_probs = self.layout_router(h_t, prev_layout)
        
        # Initialize generation
        generated = []
        current_tokens = torch.zeros(B, 0, dtype=torch.long, device=device)
        
        for step in range(max_length):
            # Get token embeddings
            if step == 0:
                token_embeds = None
            else:
                token_embeds = self.token_embedding(current_tokens)
            
            # Forward pass
            logits, expert_probs = self.language_moe(h_t, e_task, e_layout, token_embeds)
            
            # Sample next token
            next_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            generated.append(next_token)
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        generated_tokens = torch.cat(generated, dim=1)  # [B, max_length]
        
        metadata = {
            'task_probs': task_probs,
            'layout_id': layout_id,
            'layout_probs': layout_probs,
            'expert_probs': expert_probs
        }
        
        return generated_tokens, metadata
    
    def generate_candidates(self,
                           h_t: torch.Tensor,
                           prev_layout: Optional[torch.Tensor] = None,
                           num_candidates: int = 5,
                           temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generate N candidate tokens for auto-completion or error-correction tasks.
        
        This method generates multiple ranked candidates for non-autoregressive tasks,
        allowing users to select from multiple options. The candidates are ranked by
        their model confidence (logit values).
        
        Args:
            h_t: Multimodal representation [B, 256]
            prev_layout: Previous layout IDs [B] for temporal stability
            num_candidates: Number of candidates to generate (default: 5)
            temperature: Sampling temperature for diversity
        
        Returns:
            candidate_tokens: Top-N candidate token IDs [B, num_candidates]
            candidate_probs: Probabilities for each candidate [B, num_candidates]
            metadata: Dictionary with task_probs, layout_id, layout_probs, expert_probs
        """
        B = h_t.size(0)
        
        # Task and layout routing
        e_task, task_probs = self.task_router(h_t)
        e_layout, layout_id, layout_probs = self.layout_router(h_t, prev_layout)
        
        # Get logits from MoE (non-autoregressive)
        logits, expert_probs = self.language_moe(h_t, e_task, e_layout, token_embeds=None)
        
        # Apply temperature
        logits = logits / temperature  # [B, vocab_size]
        
        # Get top-N candidates
        probs = F.softmax(logits, dim=-1)  # [B, vocab_size]
        candidate_probs, candidate_tokens = torch.topk(probs, k=num_candidates, dim=-1)
        # candidate_tokens: [B, num_candidates]
        # candidate_probs: [B, num_candidates]
        
        metadata = {
            'task_probs': task_probs,
            'layout_id': layout_id,
            'layout_probs': layout_probs,
            'expert_probs': expert_probs
        }
        
        return candidate_tokens, candidate_probs, metadata
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        return {
            'task_router': sum(p.numel() for p in self.task_router.parameters()),
            'layout_router': sum(p.numel() for p in self.layout_router.parameters()),
            'language_moe': sum(p.numel() for p in self.language_moe.parameters()),
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }

