# Multi-Task Sequential Recommendation

A production-grade implementation of a multi-task sequential recommendation system trained on the Amazon Reviews 2023 (Movies & TV) dataset. The system jointly optimises for **CTR prediction** (next-item ranking) and **rating regression** using a SASRec backbone with an MMoE ranking layer, benchmarked against two baselines.

---

## Results

| Model | NDCG@10 | HR@10 | MAE |
|---|---|---|---|
| SASRec (single-task) | 0.5004 | 0.7057 | — |
| SASRec + Shared-Bottom | 0.4821 | 0.6897 | 0.9300 |
| **SASRec + MMoE** | **0.5183** | **0.7231** | **0.8924** |

> All metrics are computed on the held-out test split with 99 sampled negatives per positive.
> MMoE achieves a **3.6% improvement in NDCG@10** over the single-task SASRec baseline.

---

## Architecture

```
User interaction sequence  [i₁, i₂, …, iₜ]
         │
         ▼
┌─────────────────────────────────────────────┐
│          SASRec Encoder (shared)             │
│  Item Emb + Pos Emb                         │
│  → L × Transformer Block                    │
│     (MHSA + LayerNorm + FFN + Residual)      │
│  → h  (sequence representation)             │
└─────────────────────────────────────────────┘
         │
         ▼  (MMoE only)
┌────────────────────────────────────────────────────┐
│  K=4 Expert FFNs  {E₁, E₂, E₃, E₄}               │
│  Gate₁(h) → softmax → weighted sum → click repr    │
│  Gate₂(h) → softmax → weighted sum → rating repr   │
└────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
  Click Tower           Rating Tower
  (dot-product)         (2-layer MLP)
  BCE loss              MSE loss
```

### Models

| Model | Description |
|---|---|
| `SASRec` | Single-task transformer for next-item prediction (click only) |
| `SharedBottomMTL` | Two task heads (click + rating) sharing the same SASRec encoder |
| `MMoEMTL` | 4 expert networks + 2 task-specific gates on top of SASRec encoder |

---

## Project Structure

```
.
├── config.py                  # All hyperparameters and paths
├── requirements.txt
├── Dockerfile
├── data/
│   ├── download.py            # Download Movies_and_TV.jsonl.gz
│   ├── preprocess.py          # Parse, filter (5-core), split, save
│   └── dataset.py             # PyTorch Dataset / DataLoader classes
├── models/
│   ├── sasrec.py              # SASRec backbone + single-task model
│   ├── shared_bottom.py       # Shared-Bottom MTL model
│   └── mmoe.py                # MMoE MTL model
├── train/
│   ├── evaluate.py            # NDCG@K, HR@K, MAE metrics
│   ├── train_sasrec.py        # SASRec training + MLflow
│   ├── train_shared_bottom.py # SharedBottom training + MLflow
│   ├── train_mmoe.py          # MMoE training + MLflow
│   ├── export_onnx.py         # Export MMoE → ONNX
│   └── run_all.py             # End-to-end pipeline runner
├── serve/
│   ├── onnx_inference.py      # ONNX Runtime inference wrapper
│   └── app.py                 # FastAPI endpoints
├── notebooks/
│   └── results.ipynb          # Results comparison + plots
├── checkpoints/               # Saved model weights (created at runtime)
└── mlruns/                    # MLflow experiment data (created at runtime)
```

---

## Setup

### Prerequisites

- Python 3.10+
- ~5 GB disk space for the dataset
- (Optional) CUDA-capable GPU for faster training

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, replace the PyTorch line with the appropriate CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download and preprocess dataset

```bash
python data/download.py       # ~3.5 GB download
python data/preprocess.py     # 5-core filtering, split, negative sampling
```

### 3. Train all models

Run all three models sequentially (recommended, ~2–4 hours on GPU):

```bash
python train/run_all.py
```

Or train individually:

```bash
python train/train_sasrec.py
python train/train_shared_bottom.py
python train/train_mmoe.py
python train/export_onnx.py   # export best MMoE to ONNX
```

### 4. View experiment results in MLflow

```bash
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

### 5. Start the serving API

```bash
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
```

Test the endpoint:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"item_ids": ["B001E4KFG0", "B00813GRG4"], "top_k": 10}'
```

### 6. Open results notebook

```bash
jupyter lab notebooks/results.ipynb
```

---

## Docker

### Build

```bash
docker build -t seqrec:latest .
```

### Run

Mount the pre-built checkpoints and processed vocabulary:

```bash
docker run -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/data/processed:/app/data/processed \
    seqrec:latest
```

---

## API Reference

### `POST /recommend`

Returns top-K recommended items for a user interaction sequence.

**Request body:**
```json
{
  "item_ids": ["B001E4KFG0", "B00813GRG4", "B00KQPKJK4"],
  "top_k": 10,
  "exclude_seen": true
}
```

**Response:**
```json
{
  "recommendations": [
    {"rank": 1, "item_id": "B07XYZ1234", "internal_id": 42, "score": 3.812},
    ...
  ],
  "num_results": 10,
  "latency_ms": 12.4
}
```

### `POST /score`

Score specific candidate items.

**Request body:**
```json
{
  "item_ids": ["B001E4KFG0", "B00813GRG4"],
  "candidate_ids": ["B07XYZ1234", "B08ABC5678"]
}
```

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "num_items": 85734}
```

### `GET /info`

Returns model and catalogue metadata.

---

## Configuration

All hyperparameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `EMBED_DIM` | 64 | Item/position embedding size |
| `NUM_HEADS` | 2 | Self-attention heads per block |
| `NUM_BLOCKS` | 2 | Transformer blocks |
| `MAX_SEQ_LEN` | 50 | Input sequence length |
| `DROPOUT` | 0.2 | Dropout rate |
| `NUM_EXPERTS` | 4 | MMoE expert count |
| `EXPERT_DIM` | 128 | Expert FFN hidden size |
| `BATCH_SIZE` | 256 | Training batch size |
| `NUM_EPOCHS` | 50 | Max training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `PATIENCE` | 5 | Early-stopping patience |
| `W_CLICK` | 1.0 | Click loss weight |
| `W_RATING` | 0.5 | Rating loss weight |
| `MIN_INTERACTIONS` | 5 | 5-core filtering threshold |
| `NUM_NEG_EVAL` | 99 | Negative samples for evaluation |

---

## Implementation Notes

- **No external RecSys libraries** — all models implemented from scratch in PyTorch
- **Weight sharing**: item embeddings are shared between the encoder and the scoring layer (reduces parameters, improves generalisation)
- **Causal masking**: the self-attention blocks use an upper-triangular mask to prevent attending to future items during training
- **Negative sampling**: one random negative per positive during training; 99 sampled negatives for evaluation (standard in the literature)
- **Rating normalisation**: ratings are normalised to [0, 1] before MSE loss for training stability; outputs are clamped to [1, 5] at inference
- **Early stopping**: based on validation NDCG@10; best checkpoint is saved and used for ONNX export

---

## References

- Kang & McAuley (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). ICDM 2018.
- Ma et al. (2018). [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007). KDD 2018.
- Hou et al. (2024). [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952). Amazon Reviews 2023 dataset.
