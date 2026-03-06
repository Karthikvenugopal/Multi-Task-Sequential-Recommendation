"""
SASRec + MMoE (Mixture-of-Experts) Multi-Task Learning.

Reference:
    Jiaqi Ma et al. (2018). "Modeling Task Relationships in Multi-task Learning
    with Multi-gate Mixture-of-Experts." KDD 2018.

Architecture:
    1. SASRec encoder  →  sequence representation h  (B, d_model)
    2. K expert networks  {E_k(h)}_{k=1}^{K}  each output  (B, expert_dim)
    3. Task-specific gates:
           g_t(h) = softmax(W_t · h)   →  (B, K)
       Gated mixture:
           m_t = Σ_k  g_t[k] · E_k(h)  →  (B, expert_dim)
    4. Task-specific heads on top of m_t:
           click head  : dot-product  →  BCE loss
           rating head : MLP          →  MSE loss
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    EMBED_DIM, NUM_HEADS, NUM_BLOCKS, DROPOUT,
    MAX_SEQ_LEN, NUM_EXPERTS, EXPERT_DIM,
    W_CLICK, W_RATING,
)
from models.sasrec import SASRecEncoder
from models.shared_bottom import RatingHead


# ── Expert Network ─────────────────────────────────────────────────────────────

class ExpertNetwork(nn.Module):
    """
    A two-layer FFN expert with ReLU activation.

    Input:  (B, d_model)
    Output: (B, expert_dim)
    """

    def __init__(self, d_model: int, expert_dim: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expert_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, expert_dim)


# ── MMoE Layer ─────────────────────────────────────────────────────────────────

class MMoELayer(nn.Module):
    """
    Multi-gate Mixture-of-Experts layer.

    Given an input representation x:
      - Routes x through K experts
      - Computes K task-specific gating weights via softmax
      - Returns K task-specific mixtures
    """

    def __init__(
        self,
        d_model: int,
        expert_dim: int,
        num_experts: int,
        num_tasks: int,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(d_model, expert_dim, dropout)
            for _ in range(num_experts)
        ])

        # Task-specific gates (one gate per task)
        self.gates = nn.ModuleList([
            nn.Linear(d_model, num_experts, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(
        self, x: torch.Tensor
    ):
        """
        Args:
            x: (B, d_model)
        Returns:
            task_outputs: list of num_tasks tensors, each (B, expert_dim)
        """
        # Compute expert outputs: (B, num_experts, expert_dim)
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )

        task_outputs = []
        for gate in self.gates:
            # Gate weights: (B, num_experts)
            gate_weights = F.softmax(gate(x), dim=-1)
            # Weighted sum: (B, expert_dim)
            mixture = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
            task_outputs.append(mixture)

        return task_outputs  # [click_repr, rating_repr]


# ── Click Head (tower) ─────────────────────────────────────────────────────────

class ClickTower(nn.Module):
    """
    Task tower for click prediction on top of the gated expert mixture.

    Projects expert_dim → d_model so that we can do dot-product scoring
    against item embeddings (which live in d_model space).
    """

    def __init__(self, expert_dim: int, d_model: int, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(expert_dim, d_model),
            nn.LayerNorm(d_model, eps=1e-8),
            nn.Dropout(dropout),
        )

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        return self.proj(mixture)  # (B, d_model)


# ── MMoE Full Model ────────────────────────────────────────────────────────────

class MMoEMTL(nn.Module):
    """
    SASRec + MMoE multi-task model.

    Two tasks:
        Task 0 (click) : next-item prediction via dot-product
        Task 1 (rating): rating regression via MLP

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
        num_experts: int = NUM_EXPERTS,
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
        self.item_emb = self.encoder.item_emb  # weight sharing alias

        # MMoE layer (2 tasks: click, rating)
        self.mmoe = MMoELayer(d_model, expert_dim, num_experts, num_tasks=2, dropout=dropout)

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
        """
        Run encoder + MMoE.
        Returns (click_repr, rating_repr) both shape (B, ...).
        """
        h = self.encoder.encode_last(seq)          # (B, d_model)
        click_mix, rating_mix = self.mmoe(h)        # each (B, expert_dim)
        click_repr = self.click_tower(click_mix)    # (B, d_model)
        rating_repr = rating_mix                    # (B, expert_dim)
        return click_repr, rating_repr

    def _click_scores(
        self,
        click_repr: torch.Tensor,  # (B, d_model)
        items: torch.Tensor,       # (B,) or (B, K)
    ) -> torch.Tensor:
        if items.dim() == 1:
            e = self.item_emb(items)                              # (B, d_model)
            return (click_repr * e).sum(dim=-1)                   # (B,)
        else:
            e = self.item_emb(items)                              # (B, K, d_model)
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
    def predict_click(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """(B, K) click scores for candidate items."""
        click_repr, _ = self._encode_and_mix(seq)
        return self._click_scores(click_repr, candidates)

    @torch.no_grad()
    def predict_rating(self, seq: torch.Tensor) -> torch.Tensor:
        """(B,) predicted rating (clamped to [1, 5])."""
        _, rating_repr = self._encode_and_mix(seq)
        raw = self.rating_tower(rating_repr).squeeze(-1)
        return raw.clamp(1.0, 5.0)

    @torch.no_grad()
    def predict_both(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ):
        """
        Inference helper for the serving layer.

        Returns:
            click_scores: (B, K)
            ratings:      (B,)
        """
        click_repr, rating_repr = self._encode_and_mix(seq)
        click_scores = self._click_scores(click_repr, candidates)
        ratings = self.rating_tower(rating_repr).squeeze(-1).clamp(1.0, 5.0)
        return click_scores, ratings

    # ── ONNX export helper ─────────────────────────────────────────────────────

    def forward_onnx(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-output forward for ONNX export — returns click scores only.

        Args:
            seq:        (1, T) padded sequence
            candidates: (1, K) candidate item IDs
        Returns:
            scores: (1, K)
        """
        return self.predict_click(seq, candidates)
