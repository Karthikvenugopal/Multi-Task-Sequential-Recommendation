"""
PyTorch Dataset and DataLoader utilities for the recommendation system.

Three dataset classes:
  - SeqRecDataset      : for SASRec single-task training (click only)
  - MTLSeqRecDataset   : for multi-task training (click + rating)
  - EvalDataset        : for val / test evaluation with negative sampling
"""
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import MAX_SEQ_LEN, PROCESSED_DATA_PATH


def load_data(data_path: str = PROCESSED_DATA_PATH):
    """Load preprocessed data and vocabulary from disk."""
    import os

    with open(os.path.join(data_path, "vocab.pkl"), "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(data_path, "data.pkl"), "rb") as f:
        data = pickle.load(f)
    return vocab, data


def pad_seq(seq: List[int], max_len: int, pad: int = 0) -> List[int]:
    """Left-pad (or left-truncate) a sequence to exactly max_len."""
    seq = seq[-max_len:]
    return [pad] * (max_len - len(seq)) + seq


# ── Single-task dataset (SASRec baseline) ──────────────────────────────────────

class SeqRecDataset(Dataset):
    """
    Each sample is a (sequence, positive_item, negative_item) triple.

    For each user we slide a window over their training sequence and generate
    one training example per position where the target item exists.
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[Tuple[int, float]]],
        num_items: int,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.max_len = max_len
        self.num_items = num_items
        self.samples: List[Tuple[List[int], int]] = []  # (context, pos_item)

        for uid, seq in train_seqs.items():
            items = [item for item, _ in seq]
            for t in range(1, len(items)):
                context = items[:t]
                pos_item = items[t]
                self.samples.append((context, pos_item))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, pos_item = self.samples[idx]
        # Randomly sample one negative item
        neg_item = random.randint(1, self.num_items)
        while neg_item in set(context) or neg_item == pos_item:
            neg_item = random.randint(1, self.num_items)

        padded = pad_seq(context, self.max_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


# ── Multi-task dataset (Shared-Bottom and MMoE) ────────────────────────────────

class MTLSeqRecDataset(Dataset):
    """
    Like SeqRecDataset but also returns the rating label for the positive item.

    Returns:
        seq        : padded item sequence (context)
        pos_item   : next (positive) item id
        neg_item   : sampled negative item id
        rating     : rating of pos_item (float 1–5)
    """

    def __init__(
        self,
        train_seqs: Dict[int, List[Tuple[int, float]]],
        num_items: int,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.max_len = max_len
        self.num_items = num_items
        # (context_items, pos_item, rating)
        self.samples: List[Tuple[List[int], int, float]] = []

        for uid, seq in train_seqs.items():
            items = [item for item, _ in seq]
            ratings = [r for _, r in seq]
            for t in range(1, len(items)):
                context = items[:t]
                pos_item = items[t]
                rating = ratings[t]
                self.samples.append((context, pos_item, rating))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        context, pos_item, rating = self.samples[idx]
        neg_item = random.randint(1, self.num_items)
        while neg_item in set(context) or neg_item == pos_item:
            neg_item = random.randint(1, self.num_items)

        padded = pad_seq(context, self.max_len)
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )


# ── Evaluation dataset ─────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    """
    Dataset for evaluation (val or test).

    Each sample contains:
        seq        : padded context sequence
        target     : ground-truth positive item id
        candidates : [target] + 99 negative items  (100 items total)
        rating     : ground-truth rating for the target item
        uid        : user id (for debugging)
    """

    def __init__(
        self,
        split_seqs: Dict[int, Tuple[List[Tuple[int, float]], Tuple[int, float]]],
        negatives: Dict[int, List[int]],
        max_len: int = MAX_SEQ_LEN,
    ):
        self.max_len = max_len
        self.samples: List[Tuple[int, List[int], int, float, List[int]]] = []

        for uid, (context_seq, target_pair) in split_seqs.items():
            target_item, target_rating = target_pair
            context_items = [item for item, _ in context_seq]
            padded = pad_seq(context_items, max_len)
            neg_items = negatives.get(uid, [])[:99]
            candidates = [target_item] + neg_items  # positive always first
            self.samples.append((uid, padded, target_item, target_rating, candidates))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        uid, padded, target, rating, candidates = self.samples[idx]
        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
            torch.tensor(candidates, dtype=torch.long),
        )


# ── DataLoader helpers ─────────────────────────────────────────────────────────

def get_train_loader(
    train_seqs: Dict,
    num_items: int,
    batch_size: int,
    multitask: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if multitask:
        ds = MTLSeqRecDataset(train_seqs, num_items)
    else:
        ds = SeqRecDataset(train_seqs, num_items)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True)


def get_eval_loader(
    split_seqs: Dict,
    negatives: Dict,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataLoader:
    ds = EvalDataset(split_seqs, negatives)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)
