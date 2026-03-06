"""
Preprocessing pipeline for the Amazon Reviews 2023 dataset.

Steps:
  1. Parse raw JSONL.gz file — extract (user, item, rating, timestamp)
  2. Filter users and items with fewer than MIN_INTERACTIONS interactions
  3. Sort each user's sequence by timestamp
  4. Re-index users and items to contiguous integer IDs (1-based; 0 = padding)
  5. Split into train / val / test (leave-last-2 strategy)
  6. Save processed files + vocabulary to PROCESSED_DATA_PATH

Usage:
    python data/preprocess.py
"""
import gzip
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MIN_INTERACTIONS,
    MAX_SEQ_LEN,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
)


# ── 1. Parsing ─────────────────────────────────────────────────────────────────

def parse_raw(path: str) -> pd.DataFrame:
    """Stream-parse the JSONL.gz file into a DataFrame."""
    records: List[Dict] = []
    print(f"Parsing {path} …")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="  lines", unit=" reviews"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = obj.get("user_id") or obj.get("reviewer_id") or obj.get("userId")
            item = obj.get("parent_asin") or obj.get("asin") or obj.get("item_id")
            rating = obj.get("rating") or obj.get("overall")
            ts = (
                obj.get("timestamp")
                or obj.get("unixReviewTime")
                or obj.get("unix_time")
                or 0
            )
            if user and item and rating is not None:
                records.append(
                    {"user_id": str(user), "item_id": str(item),
                     "rating": float(rating), "timestamp": int(ts)}
                )

    df = pd.DataFrame(records)
    print(f"  Raw interactions: {len(df):,}")
    return df


# ── 2. Filtering ───────────────────────────────────────────────────────────────

def filter_kcore(df: pd.DataFrame, k: int = MIN_INTERACTIONS) -> pd.DataFrame:
    """Iteratively remove users/items with fewer than k interactions."""
    prev_len = -1
    iteration = 0
    while prev_len != len(df):
        prev_len = len(df)
        iteration += 1
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]
    print(f"  After {k}-core filtering ({iteration} iterations): {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    return df.reset_index(drop=True)


# ── 3. Re-indexing ─────────────────────────────────────────────────────────────

def build_vocab(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
    """Map raw IDs → contiguous 1-based integers. 0 is reserved for padding."""
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())

    user2id = {u: i + 1 for i, u in enumerate(users)}
    item2id = {it: i + 1 for i, it in enumerate(items)}
    id2item = {v: k for k, v in item2id.items()}
    id2user = {v: k for k, v in user2id.items()}

    num_users = len(users)
    num_items = len(items)
    print(f"  Vocabulary: {num_users:,} users, {num_items:,} items  "
          f"(IDs 1–{num_items}, 0 = PAD)")
    return user2id, item2id, id2user, id2item


# ── 4. Build sequences ─────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    user2id: Dict,
    item2id: Dict,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Returns a dict: user_int_id → list of (item_int_id, rating) sorted by time.
    """
    df = df.copy()
    df["u"] = df["user_id"].map(user2id)
    df["i"] = df["item_id"].map(item2id)
    df = df.sort_values(["u", "timestamp"])

    sequences: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        sequences[row.u].append((row.i, row.rating))

    # Deduplicate consecutive duplicates (keep latest rating if same item appears twice)
    cleaned: Dict[int, List[Tuple[int, float]]] = {}
    for uid, seq in sequences.items():
        seen = {}
        for item, rating in seq:
            seen[item] = rating  # last occurrence wins
        # But we must keep ORDER — so rebuild in order
        deduped = []
        seen_set = set()
        for item, rating in seq:
            if item not in seen_set:
                seen_set.add(item)
                deduped.append((item, rating))
        cleaned[uid] = deduped

    return cleaned


# ── 5. Train / Val / Test split ────────────────────────────────────────────────

def split_sequences(
    sequences: Dict[int, List[Tuple[int, float]]]
) -> Tuple[Dict, Dict, Dict]:
    """
    Leave-last-2 strategy:
      train: everything except last 2 items
      val:   train context + second-to-last item as target
      test:  train context + last item as target
    Users with fewer than 3 interactions are dropped.
    """
    train, val, test = {}, {}, {}
    skipped = 0
    for uid, seq in sequences.items():
        if len(seq) < 3:
            skipped += 1
            continue
        train[uid] = seq[:-2]          # all but last 2
        val[uid] = (seq[:-2], seq[-2])  # context, (item, rating)
        test[uid] = (seq[:-1], seq[-1]) # context+val_item, (item, rating)
    print(f"  Split: {len(train):,} train users, {skipped} skipped (<3 interactions)")
    return train, val, test


# ── 6. Negative sampling for evaluation ───────────────────────────────────────

def sample_negatives(
    sequences: Dict,
    num_items: int,
    num_neg: int = 99,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Sample `num_neg` negative items per user (not in their full history)."""
    rng = np.random.default_rng(seed)
    user_items: Dict[int, set] = {
        uid: set(item for item, _ in seq)
        for uid, seq in sequences.items()
    }
    negatives: Dict[int, List[int]] = {}
    all_items = np.arange(1, num_items + 1)
    for uid, item_set in tqdm(user_items.items(), desc="  neg sampling", unit=" users"):
        candidates = np.setdiff1d(all_items, list(item_set), assume_unique=True)
        neg = rng.choice(candidates, size=min(num_neg, len(candidates)), replace=False)
        negatives[uid] = neg.tolist()
    return negatives


# ── 7. Truncate / pad sequences ────────────────────────────────────────────────

def pad_or_truncate(seq: List[int], max_len: int, pad_val: int = 0) -> List[int]:
    seq = seq[-max_len:]
    return [pad_val] * (max_len - len(seq)) + seq


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # --- parse ---
    df = parse_raw(RAW_DATA_PATH)

    # --- filter ---
    df = filter_kcore(df)

    # --- vocabulary ---
    user2id, item2id, id2user, id2item = build_vocab(df)
    num_users = len(user2id)
    num_items = len(item2id)

    # --- sequences ---
    sequences = build_sequences(df, user2id, item2id)

    # --- split ---
    train_seqs, val_seqs, test_seqs = split_sequences(sequences)

    # --- negatives (based on train sequences only to avoid leakage) ---
    print("Sampling evaluation negatives …")
    val_neg = sample_negatives(
        {uid: seq for uid, seq in train_seqs.items()},
        num_items,
    )
    test_neg = sample_negatives(
        {uid: ctx + [tgt] for uid, (ctx, tgt) in val_seqs.items()},
        num_items,
    )

    # --- save ---
    vocab = {
        "user2id": user2id,
        "item2id": item2id,
        "id2user": id2user,
        "id2item": id2item,
        "num_users": num_users,
        "num_items": num_items,
    }
    with open(os.path.join(PROCESSED_DATA_PATH, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    data = {
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "test_seqs": test_seqs,
        "val_neg": val_neg,
        "test_neg": test_neg,
    }
    with open(os.path.join(PROCESSED_DATA_PATH, "data.pkl"), "wb") as f:
        pickle.dump(data, f)

    # --- stats ---
    seq_lens = [len(s) for s in train_seqs.values()]
    print(f"\nDataset statistics:")
    print(f"  Users:       {num_users:,}")
    print(f"  Items:       {num_items:,}")
    print(f"  Train users: {len(train_seqs):,}")
    print(f"  Val users:   {len(val_seqs):,}")
    print(f"  Test users:  {len(test_seqs):,}")
    print(f"  Seq len  min={min(seq_lens)} mean={np.mean(seq_lens):.1f} "
          f"median={np.median(seq_lens):.0f} max={max(seq_lens)}")
    print(f"\nSaved to {PROCESSED_DATA_PATH}/")


if __name__ == "__main__":
    main()
