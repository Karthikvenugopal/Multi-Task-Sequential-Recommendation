"""
Training script for SASRec + Shared-Bottom MTL.

Usage:
    python train/train_shared_bottom.py

MLflow experiment: multitask_seqrec
Run name: shared_bottom
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
    NUM_BLOCKS, NUM_EPOCHS, NUM_HEADS, PATIENCE, WEIGHT_DECAY, W_CLICK, W_RATING,
    PROCESSED_DATA_PATH,
)
from data.dataset import get_eval_loader, get_train_loader, load_data
from models.shared_bottom import SharedBottomMTL
from train.evaluate import evaluate_mtl


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
        train_seqs, num_items, BATCH_SIZE, multitask=True
    )
    val_loader = get_eval_loader(val_seqs, val_neg, batch_size=512)
    print(f"  Items: {num_items:,}   Train batches: {len(train_loader):,}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = SharedBottomMTL(
        num_items=num_items,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
        w_click=W_CLICK,
        w_rating=W_RATING,
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

    with mlflow.start_run(run_name="shared_bottom") as run:
        mlflow.log_params({
            "model": "SharedBottom",
            "d_model": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_blocks": NUM_BLOCKS,
            "max_len": MAX_SEQ_LEN,
            "dropout": DROPOUT,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "w_click": W_CLICK,
            "w_rating": W_RATING,
            "num_items": num_items,
            "params": param_count,
        })

        best_ndcg = -1.0
        patience_ctr = 0
        ckpt_path = os.path.join(CHECKPOINT_DIR, "shared_bottom_best.pt")

        for epoch in range(1, NUM_EPOCHS + 1):
            model.train()
            total_loss_sum = click_loss_sum = rating_loss_sum = 0.0
            t0 = time.time()

            for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False):
                seq, pos_items, neg_items, ratings = [b.to(DEVICE) for b in batch]
                optimizer.zero_grad()
                total_loss, click_loss, rating_loss = model(
                    seq, pos_items, neg_items, ratings
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss_sum += total_loss.item()
                click_loss_sum += click_loss.item()
                rating_loss_sum += rating_loss.item()

            scheduler.step()

            n_batches = len(train_loader)
            avg_total = total_loss_sum / n_batches
            avg_click = click_loss_sum / n_batches
            avg_rating = rating_loss_sum / n_batches
            elapsed = time.time() - t0

            # Validation
            val_metrics = evaluate_mtl(model, val_loader, k=10)
            ndcg = val_metrics["NDCG_at_10"]
            hr = val_metrics["HR_at_10"]
            mae = val_metrics["MAE"]

            print(
                f"Epoch {epoch:03d}  total={avg_total:.4f}  "
                f"click={avg_click:.4f}  rating={avg_rating:.4f}  "
                f"NDCG@10={ndcg:.4f}  HR@10={hr:.4f}  MAE={mae:.4f}  "
                f"({elapsed:.1f}s)"
            )

            mlflow.log_metrics(
                {
                    "train_total_loss": avg_total,
                    "train_click_loss": avg_click,
                    "train_rating_loss": avg_rating,
                    "val_NDCG_at_10": ndcg,
                    "val_HR_at_10": hr,
                    "val_MAE": mae,
                },
                step=epoch,
            )

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # ── Test ──────────────────────────────────────────────────────────────
        print("\nLoading best checkpoint for test evaluation …")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        test_seqs = data["test_seqs"]
        test_neg = data["test_neg"]
        test_loader = get_eval_loader(test_seqs, test_neg, batch_size=512)
        test_metrics = evaluate_mtl(model, test_loader, k=10)

        print("\nTest results (SharedBottom):")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
            mlflow.log_metric(f"test_{k}", v)

        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        print(f"\nMLflow run id: {run.info.run_id}")
        print(f"Best val NDCG@10: {best_ndcg:.4f}")


if __name__ == "__main__":
    train()
