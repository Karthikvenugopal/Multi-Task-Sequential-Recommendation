"""
SASRec + PLE (Progressive Layered Extraction) Multi-Task Learning.

Reference:
    Tang et al. (2020). "Progressive Layered Extraction (PLE): A Novel
    Multi-Task Learning (MTL) Model for Personalized Recommendations."
    RecSys 2020.

Motivation:
    MMoE uses a single pool of shared experts with task-specific gating.
    In practice this can cause *negative transfer* — gradients from one task
    pollute the shared expert representations for another task.

    PLE introduces two types of experts:
        • Task-specific experts  — only updated by their own task's gradient
        • Shared experts         — updated by all tasks, but attended to via
                                   per-task gating alongside task-specific experts

    Each task's gating network attends over its own specific experts PLUS the
    shared experts, giving it a private subspace while still benefiting from
    cross-task signal.

Architecture (single extraction layer, a.k.a. CGC):

    Input: h  (B, d_model)  — SASRec sequence representation

    Shared experts:     E_s^1(h), …, E_s^{K_s}(h)   → (B, expert_dim) each
    Task-1 experts:     E_1^1(h), …, E_1^{K_t}(h)   → (B, expert_dim) each
    Task-2 experts:     E_2^1(h), …, E_2^{K_t}(h)   → (B, expert_dim) each

    Gate for task t:
        candidates = [task-t experts | shared experts]   # K_t + K_s candidates
        g_t(h) = softmax(W_t · h)                        # (B, K_t + K_s)
        m_t = Σ_k g_t[k] · candidates[k]                # (B, expert_dim)

    Task towers on top of m_t  (same as MMoE):
        click tower  : Linear(expert_dim → d_model) + LayerNorm + Dropout
        rating tower : MLP(expert_dim → expert_dim//2 → 1)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBED_DIM, NUM_HEADS, NUM_BLOCKS, DROPOUT,
    MAX_SEQ_LEN, EXPERT_DIM,
    NUM_SHARED_EXPERTS, NUM_SPECIFIC_EXPERTS,
    W_CLICK, W_RATING,
)
from models.sasrec import SASRecEncoder
from models.mmoe import ExpertNetwork, ClickTower


# ── CGC Extraction Layer ────────────────────────────────────────────────────────

class CGCLayer(nn.Module):
    """
    Customized Gate Control (CGC) — the single-level extraction module of PLE.

    Each task receives a mixture drawn from its own task-specific expert pool
    plus the shared expert pool.  The shared expert pool is *not* directly
    exposed to any single task's gradient in isolation; all tasks jointly
    update it through their gating networks.
    """

    def __init__(
        self,
        d_model: int,
        expert_dim: int,
        num_shared: int,
        num_specific: int,
        num_tasks: int,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_shared = num_shared
        self.num_specific = num_specific
        self.num_tasks = num_tasks

        # Shared expert pool
        self.shared_experts = nn.ModuleList([
            ExpertNetwork(d_model, expert_dim, dropout)
            for _ in range(num_shared)
        ])

        # Per-task specific expert pools
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                ExpertNetwork(d_model, expert_dim, dropout)
                for _ in range(num_specific)
            ])
            for _ in range(num_tasks)
        ])

        # Per-task gating networks
        # Each gate attends over (num_specific + num_shared) experts
        num_candidates = num_specific + num_shared
        self.gates = nn.ModuleList([
            nn.Linear(d_model, num_candidates, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, d_model)
        Returns:
            List of num_tasks tensors, each (B, expert_dim)
        """
        # Shared expert outputs: (num_shared, B, expert_dim)
        shared_outs = [e(x) for e in self.shared_experts]

        task_outputs = []
        for t, (task_expert_pool, gate) in enumerate(
            zip(self.task_experts, self.gates)
        ):
            # Task-specific expert outputs
            specific_outs = [e(x) for e in task_expert_pool]

            # Concatenate [task-specific | shared] along expert dim
            # (num_specific + num_shared, B, expert_dim)
            all_outs = torch.stack(specific_outs + shared_outs, dim=1)  # (B, K_t+K_s, expert_dim)

            # Gate weights over all candidate experts
            gate_weights = F.softmax(gate(x), dim=-1)  # (B, K_t+K_s)

            # Weighted mixture
            mixture = (gate_weights.unsqueeze(-1) * all_outs).sum(dim=1)  # (B, expert_dim)
            task_outputs.append(mixture)

        return task_outputs  # [click_repr, rating_repr]


# ── PLE Full Model ──────────────────────────────────────────────────────────────

class PLEMTLModel(nn.Module):
    """
    SASRec + PLE (single extraction level) multi-task model.

    Two tasks:
        Task 0 (click) : next-item prediction via dot-product → BCE loss
        Task 1 (rating): rating regression via MLP → MSE loss

    Training loss:
        L = w_click * L_click + w_rating * L_rating
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_blocks: int = NUM_BLOCKS,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
        num_shared: int = NUM_SHARED_EXPERTS,
        num_specific: int = NUM_SPECIFIC_EXPERTS,
        expert_dim: int = EXPERT_DIM,
        w_click: float = W_CLICK,
        w_rating: float = W_RATING,
    ):
        super().__init__()
        self.w_click = w_click
        self.w_rating = w_rating
        self.d_model = d_model

        # Shared SASRec encoder
        self.encoder = SASRecEncoder(
            num_items, d_model, num_heads, num_blocks, max_len, dropout
        )
        self.item_emb = self.encoder.item_emb  # weight-sharing alias

        # CGC extraction layer (PLE)
        self.cgc = CGCLayer(
            d_model=d_model,
            expert_dim=expert_dim,
            num_shared=num_shared,
            num_specific=num_specific,
            num_tasks=2,
            dropout=dropout,
        )

        # Task towers
        self.click_tower = ClickTower(expert_dim, d_model, dropout)
        self.rating_tower = nn.Sequential(
            nn.Linear(expert_dim, expert_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim // 2, 1),
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _encode_and_mix(self, seq: torch.Tensor):
        h = self.encoder.encode_last(seq)           # (B, d_model)
        click_mix, rating_mix = self.cgc(h)         # each (B, expert_dim)
        click_repr = self.click_tower(click_mix)    # (B, d_model)
        return click_repr, rating_mix

    def _click_scores(self, click_repr: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if items.dim() == 1:
            e = self.item_emb(items)                               # (B, d_model)
            return (click_repr * e).sum(dim=-1)                    # (B,)
        else:
            e = self.item_emb(items)                               # (B, K, d_model)
            return torch.bmm(e, click_repr.unsqueeze(-1)).squeeze(-1)  # (B, K)

    # ── Training forward ───────────────────────────────────────────────────────

    def forward(
        self,
        seq: torch.Tensor,        # (B, T)
        pos_items: torch.Tensor,  # (B,)
        neg_items: torch.Tensor,  # (B,)
        ratings: torch.Tensor,    # (B,)
    ):
        """Return (total_loss, click_loss, rating_loss)."""
        click_repr, rating_repr = self._encode_and_mix(seq)

        # Click loss (BCE)
        pos_scores = self._click_scores(click_repr, pos_items)
        neg_scores = self._click_scores(click_repr, neg_items)
        click_loss = (
            F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        ) / 2

        # Rating loss (MSE on normalised ratings)
        pred_rating = self.rating_tower(rating_repr).squeeze(-1)  # (B,)
        norm_true = (ratings - 1.0) / 4.0
        norm_pred = (pred_rating - 1.0) / 4.0
        rating_loss = F.mse_loss(norm_pred, norm_true)

        total = self.w_click * click_loss + self.w_rating * rating_loss
        return total, click_loss, rating_loss

    # ── Inference ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_click(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """(B, K) click scores for candidate items."""
        click_repr, _ = self._encode_and_mix(seq)
        return self._click_scores(click_repr, candidates)

    @torch.no_grad()
    def predict_rating(self, seq: torch.Tensor) -> torch.Tensor:
        """(B,) predicted rating (clamped to [1, 5])."""
        _, rating_repr = self._encode_and_mix(seq)
        return self.rating_tower(rating_repr).squeeze(-1).clamp(1.0, 5.0)

    @torch.no_grad()
    def predict_both(self, seq: torch.Tensor, candidates: torch.Tensor):
        click_repr, rating_repr = self._encode_and_mix(seq)
        click_scores = self._click_scores(click_repr, candidates)
        ratings = self.rating_tower(rating_repr).squeeze(-1).clamp(1.0, 5.0)
        return click_scores, ratings
