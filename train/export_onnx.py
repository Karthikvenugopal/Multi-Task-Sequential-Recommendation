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
    # Use non-zero sequence so softmax doesn't NaN on fully-padded input
    dummy_seq = torch.arange(1, MAX_SEQ_LEN + 1, dtype=torch.long).unsqueeze(0) % (num_items + 1)
    dummy_seq[dummy_seq == 0] = 1
    dummy_candidates = torch.arange(1, K + 1, dtype=torch.long).unsqueeze(0)

    # Verify the model runs
    with torch.no_grad():
        out = model.forward_onnx(dummy_seq, dummy_candidates)
    print(f"  Test forward OK — output shape: {out.shape}")

    # ── Patch attention blocks for ONNX compatibility ──────────────────────────
    # nn.MultiheadAttention uses _native_multi_head_attention (fused kernel)
    # which is not ONNX-serializable. Replace with a manual implementation
    # using only primitive ops that ONNX can handle.
    import types
    import torch.nn.functional as F

    def _patched_sasrec_block_forward(self, x, key_padding_mask):
        B, T, d_model = x.shape
        num_heads = self.attn.num_heads
        head_dim = d_model // num_heads
        scale = head_dim ** -0.5

        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )

        # Self-attention sub-layer
        residual = x
        x = self.norm1(x)

        # Manual QKV projection and multi-head attention
        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, num_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        # Apply causal mask
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0).float() * -1e9
        # Apply key padding mask
        scores = scores + key_padding_mask.unsqueeze(1).unsqueeze(2).float() * -1e9

        attn_w = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_w, v)                               # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T, d_model)  # (B, T, d_model)
        out = F.linear(out, self.attn.out_proj.weight, self.attn.out_proj.bias)

        x = self.dropout(out) + residual

        # FFN sub-layer
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x

    for block in model.encoder.blocks:
        block.forward = types.MethodType(_patched_sasrec_block_forward, block)

    # ── Export ─────────────────────────────────────────────────────────────────
    # Wrap model so legacy exporter calls forward_onnx instead of forward
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, seq, candidates):
            return self.m.forward_onnx(seq, candidates)

    print(f"Exporting to {onnx_path} …")
    torch.onnx.export(
        OnnxWrapper(model),
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
        dynamo=False,
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
