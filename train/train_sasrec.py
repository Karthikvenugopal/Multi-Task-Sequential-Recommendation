"""
Training script for SASRec (single-task, click prediction baseline).

Usage:
    python train/train_sasrec.py

MLflow experiment: multitask_seqrec
Run name: sasrec
"""
import os
import sys
import time
from pathlib import Path

import mlflow
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BATCH_SIZE, CHECKPOINT_DIR, DEVICE, EMBED_DIM, DROPOUT,
    LEARNING_RATE, MAX_SEQ_LEN, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
    NUM_BLOCKS, NUM_EPOCHS, NUM_HEADS, PATIENCE, WEIGHT_DECAY,
    PROCESSED_DATA_PATH,
)
from data.dataset import get_eval_loader, get_train_loader, load_data
from models.sasrec import SASRec
from train.evaluate import evaluate_sasrec


def train() -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    vocab, data = load_data(PROCESSED_DATA_PATH)
    num_items = vocab["num_items"]
    train_seqs = data["train_seqs"]
    val_seqs = data["val_seqs"]
    val_neg = data["val_neg"]

    train_loader = get_train_loader(
        train_seqs, num_items, BATCH_SIZE, multitask=False
    )
    val_loader = get_eval_loader(val_seqs, val_neg, batch_size=512)

    print(f"  Items: {num_items:,}   Train batches: {len(train_loader):,}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = SASRec(
        num_items=num_items,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count:,}")

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="sasrec") as run:
        mlflow.log_params({
            "model": "SASRec",
            "d_model": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "max_len": MAX_SEQ_LEN,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "num_items": num_items,
            "params": param_count,
        })

        # ── Training loop ──────────────────────────────────────────────────────
        best_ndcg = -1.0
        patience_ctr = 0
        ckpt_path = os.path.join(CHECKPOINT_DIR, "sasrec_best.pt")

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
                seq, pos_items, neg_items = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                loss = model(seq, pos_items, neg_items)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            avg_loss = epoch_loss / len(train_loader)
            elapsed = time.time() - t0

            # Validation
            val_metrics = evaluate_sasrec(model, val_loader, k=10)
            ndcg = val_metrics["NDCG_at_10"]
            hr = val_metrics["HR_at_10"]

            print(
                f"Epoch {epoch:03d}  loss={avg_loss:.4f}  "
                f"NDCG@10={ndcg:.4f}  HR@10={hr:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # Log to MLflow
            mlflow.log_metrics(
                {"train_loss": avg_loss, "val_NDCG_at_10": ndcg, "val_HR_at_10": hr},
                step=epoch,
            )

            # Early stopping & checkpointing
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                    break

        # ── Final test evaluation ──────────────────────────────────────────────
        print("\nLoading best checkpoint for test evaluation …")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        test_seqs = data["test_seqs"]
        test_neg = data["test_neg"]
        test_loader = get_eval_loader(test_seqs, test_neg, batch_size=512)
        test_metrics = evaluate_sasrec(model, test_loader, k=10)

        print("\nTest results (SASRec):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            mlflow.log_metric(f"test_{k}", v)

        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        print(f"\nMLflow run id: {run.info.run_id}")
        print(f"Best val NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    train()
