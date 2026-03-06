"""
SASRec + Shared-Bottom Multi-Task Learning.

Architecture:
    Shared SASRec encoder  (same backbone for both tasks)
        ├── Click head     : dot-product → BCE loss
        └── Rating head    : MLP → scalar rating prediction → MSE loss

The "shared bottom" refers to the fact that both task heads receive the
*same* sequence representation from the encoder without any task-specific
mixture or gating.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBED_DIM, NUM_HEADS, NUM_BLOCKS, DROPOUT, MAX_SEQ_LEN, W_CLICK, W_RATING
from models.sasrec import SASRecEncoder


class RatingHead(nn.Module):
    """
    Two-layer MLP that maps a sequence representation to a scalar rating.

    Input:  (B, d_model) — sequence representation
    Output: (B,)         — predicted rating (raw logit; clamp to [1, 5] at inference)
    """

    def __init__(self, d_model: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)  # (B,)


class SharedBottomMTL(nn.Module):
    """
    Shared-Bottom Multi-Task SASRec.

    Training loss:
        L = w_click * L_click + w_rating * L_rating

    where
        L_click  = BCE on (pos, neg) item scores
        L_rating = MSE on predicted vs true rating for the positive item
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_blocks: int = NUM_BLOCKS,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
        w_click: float = W_CLICK,
        w_rating: float = W_RATING,
    ):
        super().__init__()
        self.w_click = w_click
        self.w_rating = w_rating

        # Shared encoder
        self.encoder = SASRecEncoder(
            num_items, d_model, num_heads, num_blocks, max_len, dropout
        )
        self.item_emb = self.encoder.item_emb  # shared weights alias

        # Task-specific heads
        self.rating_head = RatingHead(d_model, dropout)

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _click_scores(
        self, h: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """
        Dot-product click score.
        h:     (B, d_model)
        items: (B,) or (B, K)
        """
        if items.dim() == 1:
            e = self.item_emb(items)                              # (B, d_model)
            return (h * e).sum(dim=-1)                            # (B,)
        else:
            e = self.item_emb(items)                              # (B, K, d_model)
            return torch.bmm(e, h.unsqueeze(-1)).squeeze(-1)      # (B, K)

    # ── Forward / loss ─────────────────────────────────────────────────────────

    def forward(
        self,
        seq: torch.Tensor,        # (B, T)
        pos_items: torch.Tensor,  # (B,)
        neg_items: torch.Tensor,  # (B,)
        ratings: torch.Tensor,    # (B,)  true rating for pos_item
    ):
        """Return (total_loss, click_loss, rating_loss)."""
        h = self.encoder.encode_last(seq)  # (B, d_model)

        # Click task (BPR-style BCE)
        pos_scores = self._click_scores(h, pos_items)
        neg_scores = self._click_scores(h, neg_items)
        click_loss = (
            F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            + F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        ) / 2

        # Rating task
        pred_ratings = self.rating_head(h)                        # (B,)
        # Normalise ratings to [0, 1] for stable MSE training
        norm_true = (ratings - 1.0) / 4.0
        norm_pred = (pred_ratings - 1.0) / 4.0
        rating_loss = F.mse_loss(norm_pred, norm_true)

        total = self.w_click * click_loss + self.w_rating * rating_loss
        return total, click_loss, rating_loss

    # ── Prediction ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_click(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """(B, K) click scores for candidate items."""
        h = self.encoder.encode_last(seq)
        return self._click_scores(h, candidates)

    @torch.no_grad()
    def predict_rating(self, seq: torch.Tensor) -> torch.Tensor:
        """(B,) predicted rating for the next interacted item."""
        h = self.encoder.encode_last(seq)
        raw = self.rating_head(h)
        return raw.clamp(1.0, 5.0)
