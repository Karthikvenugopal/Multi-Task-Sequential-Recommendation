"""
Evaluation utilities for ranking and rating metrics.

Metrics:
    - NDCG@K  (Normalised Discounted Cumulative Gain)
    - Hit Rate@K (= Recall@K when there is one positive per user)
    - MAE      (Mean Absolute Error for rating prediction)
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE


# ── Metric helpers ─────────────────────────────────────────────────────────────

def ndcg_at_k(ranked_items: List[int], target: int, k: int = 10) -> float:
    """NDCG@k for a single query (one positive item, rest negatives)."""
    for rank, item in enumerate(ranked_items[:k]):
        if item == target:
            return 1.0 / math.log2(rank + 2)  # +2 because rank is 0-indexed
    return 0.0


def hit_at_k(ranked_items: List[int], target: int, k: int = 10) -> float:
    """Hit@k (1 if target in top-k, else 0)."""
    return float(target in ranked_items[:k])


# ── Single-task evaluation (SASRec) ───────────────────────────────────────────

@torch.no_grad()
def evaluate_sasrec(
    model: nn.Module,
    loader: DataLoader,
    k: int = 10,
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate SASRec (single-task) on a DataLoader.

    Each batch from EvalDataset contains:
        seq        : (B, T)
        target     : (B,)
        rating     : (B,)   — unused for single-task
        candidates : (B, C) — first column is the positive item
    """
    model.eval()
    ndcg_sum, hit_sum, n = 0.0, 0.0, 0

    for seq, target, rating, candidates in loader:
        seq = seq.to(device)
        candidates = candidates.to(device)
        target = target.to(device)

        scores = model.predict(seq, candidates)  # (B, C)
        # Sort candidates by descending score
        _, order = scores.sort(dim=-1, descending=True)
        ranked = candidates.gather(dim=1, index=order)  # (B, C) re-ordered

        B = seq.size(0)
        for b in range(B):
            t = target[b].item()
            ranked_list = ranked[b].tolist()
            ndcg_sum += ndcg_at_k(ranked_list, t, k)
            hit_sum += hit_at_k(ranked_list, t, k)
        n += B

    return {
        f"NDCG_at_{k}": ndcg_sum / n,
        f"HR_at_{k}": hit_sum / n,
    }


# ── Multi-task evaluation (SharedBottom / MMoE) ────────────────────────────────

@torch.no_grad()
def evaluate_mtl(
    model: nn.Module,
    loader: DataLoader,
    k: int = 10,
    device: str = DEVICE,
) -> Dict[str, float]:
    """
    Evaluate an MTL model on a DataLoader.

    Returns NDCG@k, HR@k (click task) and MAE (rating task).
    """
    model.eval()
    ndcg_sum, hit_sum, mae_sum, n = 0.0, 0.0, 0.0, 0

    for seq, target, rating, candidates in loader:
        seq = seq.to(device)
        candidates = candidates.to(device)
        target = target.to(device)
        rating = rating.to(device)

        # Click ranking
        click_scores = model.predict_click(seq, candidates)  # (B, C)
        _, order = click_scores.sort(dim=-1, descending=True)
        ranked = candidates.gather(dim=1, index=order)       # (B, C)

        # Rating prediction (for the next item — use sequence representation)
        pred_ratings = model.predict_rating(seq)             # (B,)
        mae_sum += (pred_ratings - rating).abs().sum().item()

        B = seq.size(0)
        for b in range(B):
            t = target[b].item()
            ranked_list = ranked[b].tolist()
            ndcg_sum += ndcg_at_k(ranked_list, t, k)
            hit_sum += hit_at_k(ranked_list, t, k)
        n += B

    return {
        f"NDCG_at_{k}": ndcg_sum / n,
        f"HR_at_{k}": hit_sum / n,
        "MAE": mae_sum / n,
    }
