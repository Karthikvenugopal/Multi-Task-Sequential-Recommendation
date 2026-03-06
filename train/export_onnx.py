"""
Export the best MMoE model checkpoint to ONNX format.

The exported model:
    Inputs:
        seq        : int64 (1, MAX_SEQ_LEN)   — padded item sequence
        candidates : int64 (1, K)              — candidate item IDs

    Output:
        scores     : float32 (1, K)            — click scores for candidates

Usage:
    python train/export_onnx.py
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BEST_MMOE_CKPT, CHECKPOINT_DIR, DEVICE, EMBED_DIM, DROPOUT,
    EXPERT_DIM, MAX_SEQ_LEN, NUM_BLOCKS, NUM_EXPERTS, NUM_HEADS,
    ONNX_MODEL_PATH, PROCESSED_DATA_PATH, W_CLICK, W_RATING,
)
from data.dataset import load_data
from models.mmoe import MMoEMTL


def export(onnx_path: str = ONNX_MODEL_PATH) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load vocabulary ────────────────────────────────────────────────────────
    print("Loading vocabulary …")
    vocab, _ = load_data(PROCESSED_DATA_PATH)
    num_items = vocab["num_items"]
    print(f"  num_items = {num_items:,}")

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading MMoE checkpoint from {BEST_MMOE_CKPT} …")
    model = MMoEMTL(
        num_items=num_items,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        num_experts=NUM_EXPERTS,
        expert_dim=EXPERT_DIM,
        w_click=W_CLICK,
        w_rating=W_RATING,
    )
    model.load_state_dict(torch.load(BEST_MMOE_CKPT, map_location="cpu"))
    model.eval()

    # ── Dummy inputs ───────────────────────────────────────────────────────────
    K = 100  # number of candidates (must match serving layer)
    dummy_seq = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long)
    dummy_candidates = torch.arange(1, K + 1, dtype=torch.long).unsqueeze(0)

    # Verify the model runs
    with torch.no_grad():
        out = model.forward_onnx(dummy_seq, dummy_candidates)
    print(f"  Test forward OK — output shape: {out.shape}")

    # ── Export ─────────────────────────────────────────────────────────────────
    print(f"Exporting to {onnx_path} …")
    torch.onnx.export(
        model,
        args=(dummy_seq, dummy_candidates),
        f=onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["seq", "candidates"],
        output_names=["scores"],
        dynamic_axes={
            "seq":        {0: "batch_size"},
            "candidates": {0: "batch_size", 1: "num_candidates"},
            "scores":     {0: "batch_size", 1: "num_candidates"},
        },
    )

    # ── Verify with ONNX Runtime ───────────────────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            None,
            {
                "seq": dummy_seq.numpy(),
                "candidates": dummy_candidates.numpy(),
            },
        )[0]
        max_diff = float(np.abs(out.numpy() - ort_out).max())
        print(f"  ONNX Runtime verification — max diff vs PyTorch: {max_diff:.2e}")
        if max_diff < 1e-4:
            print("  ✓ ONNX export verified successfully!")
        else:
            print("  ⚠ Large numerical difference detected — check model/opset.")
    except ImportError:
        print("  onnxruntime not installed — skipping verification.")

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"\nSaved ONNX model ({size_mb:.1f} MB) to:\n  {onnx_path}")


if __name__ == "__main__":
    export()
