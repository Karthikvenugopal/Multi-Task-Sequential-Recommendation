"""
ONNX Runtime inference wrapper for the MMoE model.

Provides:
    OnnxRecommender.recommend(item_ids, top_k) → list of (item_id, score)
"""
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from config import MAX_SEQ_LEN, ONNX_MODEL_PATH, VOCAB_PATH


class OnnxRecommender:
    """
    Stateless inference wrapper that loads the ONNX model once at startup
    and serves recommendations without any PyTorch dependency.

    Attributes:
        num_items (int): total number of items in the catalogue
        item2id   (dict): raw item-string → internal integer ID
        id2item   (dict): internal integer ID → raw item-string
    """

    def __init__(
        self,
        onnx_path: str = ONNX_MODEL_PATH,
        vocab_path: str = VOCAB_PATH,
        providers: Optional[List[str]] = None,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        self.item2id: Dict[str, int] = vocab["item2id"]
        self.id2item: Dict[int, str] = vocab["id2item"]
        self.num_items: int = vocab["num_items"]
        self.max_len: int = MAX_SEQ_LEN

        # Load ONNX session
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=so,
            providers=providers,
        )
        # Warm up
        dummy_seq = np.zeros((1, MAX_SEQ_LEN), dtype=np.int64)
        dummy_cands = np.arange(1, 11, dtype=np.int64).reshape(1, -1)
        self._session.run(None, {"seq": dummy_seq, "candidates": dummy_cands})

    # ── Encoding helpers ───────────────────────────────────────────────────────

    def _encode_sequence(self, item_ids: List[str]) -> np.ndarray:
        """
        Convert a list of raw item-string IDs to a left-padded integer array.

        Unknown items are silently skipped.
        """
        int_ids = [
            self.item2id[iid]
            for iid in item_ids
            if iid in self.item2id
        ]
        # Truncate to max_len, then left-pad with zeros
        int_ids = int_ids[-self.max_len:]
        padded = [0] * (self.max_len - len(int_ids)) + int_ids
        return np.array(padded, dtype=np.int64).reshape(1, -1)

    def _build_candidates(
        self, exclude: Optional[set] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build a candidate array of all catalogue items, excluding the user's
        own history (to avoid re-recommending already-seen items).
        """
        if exclude is None:
            exclude = set()
        cand_ids = [i for i in range(1, self.num_items + 1) if i not in exclude]
        return np.array(cand_ids, dtype=np.int64).reshape(1, -1), cand_ids

    # ── Main inference ─────────────────────────────────────────────────────────

    def recommend(
        self,
        item_ids: List[str],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> List[Dict]:
        """
        Return the top-k recommended items for a user interaction sequence.

        Args:
            item_ids:     List of raw item identifiers (strings) representing
                          the user's interaction history (chronological order).
            top_k:        Number of items to return.
            exclude_seen: If True, exclude already-interacted items.

        Returns:
            List of dicts, each with keys:
                "item_id"    : str   — original item identifier
                "internal_id": int   — internal integer ID
                "score"      : float — click probability score (logit)
                "rank"       : int   — 1-based rank
        """
        # Encode sequence
        seq = self._encode_sequence(item_ids)

        # Build candidate set
        exclude = set()
        if exclude_seen:
            exclude = {self.item2id[iid] for iid in item_ids if iid in self.item2id}

        cand_arr, cand_ids = self._build_candidates(exclude)

        # Run ONNX inference in batches to avoid OOM on huge catalogues
        BATCH = 4096
        all_scores = []
        for start in range(0, len(cand_ids), BATCH):
            end = start + BATCH
            batch_cands = cand_arr[:, start:end]
            scores_batch = self._session.run(
                None, {"seq": seq, "candidates": batch_cands}
            )[0][0]  # (batch,)
            all_scores.extend(scores_batch.tolist())

        # Top-k
        scored = sorted(
            zip(cand_ids, all_scores), key=lambda x: x[1], reverse=True
        )
        top = scored[:top_k]

        return [
            {
                "rank": rank + 1,
                "item_id": self.id2item.get(iid, str(iid)),
                "internal_id": iid,
                "score": float(score),
            }
            for rank, (iid, score) in enumerate(top)
        ]

    # ── Batch inference (for testing / evaluation) ─────────────────────────────

    def score_candidates(
        self,
        item_ids: List[str],
        candidate_ids: List[str],
    ) -> List[float]:
        """
        Score a specific set of candidate items (identified by raw string IDs).

        Returns a list of float scores aligned with `candidate_ids`.
        """
        seq = self._encode_sequence(item_ids)
        cand_ints = np.array(
            [self.item2id.get(c, 0) for c in candidate_ids], dtype=np.int64
        ).reshape(1, -1)
        scores = self._session.run(
            None, {"seq": seq, "candidates": cand_ints}
        )[0][0]
        return scores.tolist()
