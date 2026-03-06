"""
SASRec — Self-Attentive Sequential Recommendation (single-task).

Reference:
    Wang-Cheng Kang and Julian McAuley (2018).
    "Self-Attentive Sequential Recommendation." ICDM 2018.

Architecture:
    Item Embedding + Positional Embedding
    → L × (LayerNorm → Multi-Head Self-Attention → Dropout → Residual
           → LayerNorm → FFN → Dropout → Residual)
    → Final position representation
    → Dot-product score with candidate item embeddings

The same Encoder is reused by SharedBottomMTL and MMoEMTL.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EMBED_DIM, NUM_HEADS, NUM_BLOCKS, DROPOUT, MAX_SEQ_LEN


# ── Point-wise Feed-Forward Network ───────────────────────────────────────────

class PointWiseFeedForward(nn.Module):
    """Two-layer FFN with GELU activation, applied position-wise."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Transformer Block ──────────────────────────────────────────────────────────

class SASRecBlock(nn.Module):
    """
    One SASRec transformer block:
        LayerNorm → MHSA → Dropout → Residual
        LayerNorm → FFN  → Dropout → Residual
    """

    def __init__(
        self,
        d_model: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-8)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-8)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = PointWiseFeedForward(d_model, d_ff=d_model * 4, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # (B, T, d_model)
        key_padding_mask: torch.Tensor,  # (B, T) — True where padded
    ) -> torch.Tensor:
        # Causal (look-ahead) mask: upper-triangular (excluding diagonal)
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Self-attention sub-layer
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.dropout(x) + residual

        # FFN sub-layer
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


# ── SASRec Encoder (shared backbone) ──────────────────────────────────────────

class SASRecEncoder(nn.Module):
    """
    Shared SASRec backbone that converts a padded item sequence into
    a dense sequence of hidden states.

    Output: tensor of shape (B, T, d_model) — one vector per time step.
    The representation used for prediction is typically the last non-padded
    position, exposed via `encode_last(seq)`.
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_blocks: int = NUM_BLOCKS,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = 0  # 0 is the padding token

        # Item and positional embeddings (item IDs are 1-based; 0 = PAD)
        self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)  # position 0 unused
        self.emb_dropout = nn.Dropout(dropout)
        self.emb_norm = nn.LayerNorm(d_model, eps=1e-8)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, T) padded item IDs (0 = PAD, left-padded)
        Returns:
            hidden: (B, T, d_model)
        """
        B, T = seq.shape
        # Positional indices 1..T (0 reserved for unused)
        positions = torch.arange(1, T + 1, device=seq.device).unsqueeze(0).expand(B, T)

        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.emb_dropout(self.emb_norm(x))

        # Padding mask: True where the token is PAD
        key_padding_mask = seq == self.pad_id  # (B, T)

        for block in self.blocks:
            x = block(x, key_padding_mask)

        return x  # (B, T, d_model)

    def encode_last(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Returns the hidden state at the last *non-padded* position for each
        sequence. Shape: (B, d_model).
        """
        hidden = self.forward(seq)  # (B, T, d_model)
        # Index of the last non-pad token (right-most 1 in non-pad mask)
        non_pad = (seq != self.pad_id).long()           # (B, T)
        last_idx = non_pad.cumsum(dim=1).argmax(dim=1)  # (B,) index of last non-pad
        # Gather the corresponding hidden states
        last_hidden = hidden[torch.arange(hidden.size(0), device=seq.device), last_idx]
        return last_hidden  # (B, d_model)


# ── SASRec (single-task model) ─────────────────────────────────────────────────

class SASRec(nn.Module):
    """
    SASRec for next-item prediction only (click task).

    Training:
        Binary cross-entropy on (positive, negative) item pairs.
        score(seq, item) = dot(encoder(seq), item_emb(item))

    Inference:
        Returns scores for all candidate items, used for ranking.
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_blocks: int = NUM_BLOCKS,
        max_len: int = MAX_SEQ_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.encoder = SASRecEncoder(
            num_items, d_model, num_heads, num_blocks, max_len, dropout
        )
        # Item embedding weights are shared with the encoder for score computation
        self.item_emb = self.encoder.item_emb

    def score(
        self, seq: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dot-product scores for given items.

        Args:
            seq:   (B, T) padded sequence
            items: (B,) or (B, K) item IDs
        Returns:
            scores: (B,) or (B, K)
        """
        h = self.encoder.encode_last(seq)    # (B, d_model)
        if items.dim() == 1:
            e = self.item_emb(items)          # (B, d_model)
            return (h * e).sum(dim=-1)        # (B,)
        else:
            e = self.item_emb(items)          # (B, K, d_model)
            return torch.bmm(e, h.unsqueeze(-1)).squeeze(-1)  # (B, K)

    def forward(
        self,
        seq: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ):
        """
        Compute BPR/BCE training loss.

        Args:
            seq:       (B, T) padded context
            pos_items: (B,) positive item IDs
            neg_items: (B,) negative item IDs
        Returns:
            loss: scalar
        """
        pos_scores = self.score(seq, pos_items)   # (B,)
        neg_scores = self.score(seq, neg_items)   # (B,)

        # Binary cross-entropy loss (positive = 1, negative = 0)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        return (pos_loss + neg_loss) / 2

    @torch.no_grad()
    def predict(
        self, seq: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        """
        Rank candidates for evaluation.

        Args:
            seq:        (B, T)
            candidates: (B, K) candidate item IDs
        Returns:
            scores: (B, K)
        """
        return self.score(seq, candidates)
