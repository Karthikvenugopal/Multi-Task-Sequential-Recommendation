# Multi-Task Sequential Recommendation

A production-grade implementation of a multi-task sequential recommendation system trained on the Amazon Reviews 2023 (Movies & TV) dataset. The system jointly optimises for **CTR prediction** (next-item ranking) and **rating regression** using a SASRec backbone, benchmarked across three MTL architectures: Shared-Bottom, MMoE, and PLE.

This project mirrors the core recommendation stack used at short-video and content platforms: a transformer-based sequence encoder, multi-task learning to handle heterogeneous signals (engagement vs. satisfaction), and ONNX-exported real-time serving via FastAPI.

---

## Results

| Model | NDCG@10 | HR@10 | MAE | Val NDCG@10 |
|---|---|---|---|---|
| SASRec (single-task) | 0.5004 | 0.7057 | — | — |
| SASRec + Shared-Bottom | 0.5275 | 0.7349 | 0.9635 | 0.5693 |
| SASRec + MMoE | 0.5111 | 0.7191 | 0.9546 | 0.5487 |
| **SASRec + PLE** | **0.5156** | **0.7227** | **0.9572** | **0.5549** |

> All metrics evaluated on the held-out test split with 99 sampled negatives per positive.

**Key findings:** SharedBottom remains the strongest model (NDCG@10 0.5275). PLE outperforms MMoE (+0.45pp NDCG@10) as expected — task-specific experts reduce negative transfer — but does not surpass SharedBottom (-1.19pp). The val→test gaps are similar across all models (~0.04), ruling out overfitting. The likely explanation is dataset scale: Amazon Movies & TV is relatively small for PLE's additional parameters, and SharedBottom's hard parameter sharing acts as an implicit regularizer on sparse data.

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
│     (Causal MHSA + LayerNorm + FFN)         │
│  → h  (B, d_model)                          │
└─────────────────────────────────────────────┘
         │
         ├──── Shared-Bottom ────────────────────────────────────────────┐
         │     (same h for both tasks)                                   │
         │                                                               │
         ├──── MMoE ─────────────────────────────────────────────────────┤
         │     K shared experts, 2 task-specific gates                   │
         │     gate_t(h) = softmax(W_t · h)  →  mixture_t               │
         │                                                               │
         └──── PLE (CGC) ─────────────────────────────────────────────── ┘
               K_s shared experts  +  K_t task-specific experts per task
               gate_t(h) attends over [task-t experts | shared experts]
               → reduced negative transfer, task-private gradient subspace

         │                    │
         ▼                    ▼
  Click Tower           Rating Tower
  Linear(expert_dim     MLP(expert_dim
  → d_model)            → expert_dim//2 → 1)
  dot-product score     MSE loss
  BCE loss
```

### Model Comparison

| Model | Expert sharing | Task isolation | Neg. transfer risk |
|---|---|---|---|
| SharedBottom | Full (encoder) | None | Low (no expert routing) |
| MMoE | All experts shared | Via gate weights | Medium |
| PLE | Shared + task-specific | Task-specific pool | Low |

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
│   ├── mmoe.py                # MMoE MTL model
│   └── ple.py                 # PLE (CGC) MTL model
├── train/
│   ├── evaluate.py            # NDCG@K, HR@K, MAE metrics
│   ├── train_sasrec.py        # SASRec training + MLflow
│   ├── train_shared_bottom.py # SharedBottom training + MLflow
│   ├── train_mmoe.py          # MMoE training + MLflow
│   ├── train_ple.py           # PLE training + MLflow
│   ├── export_onnx.py         # Export best model → ONNX
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
- (Optional) CUDA-capable GPU or Apple Silicon (MPS) for faster training

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

Run all models sequentially (recommended):

```bash
python train/run_all.py
```

Or train individually:

```bash
python train/train_sasrec.py
python train/train_shared_bottom.py
python train/train_mmoe.py
python train/train_ple.py
python train/export_onnx.py   # export best model to ONNX
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

Score specific candidate items for re-ranking.

**Request body:**
```json
{
  "item_ids": ["B001E4KFG0", "B00813GRG4"],
  "candidate_ids": ["B07XYZ1234", "B08ABC5678"]
}
```

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "num_items": 200151}
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
| `NUM_SHARED_EXPERTS` | 2 | PLE shared expert count |
| `NUM_SPECIFIC_EXPERTS` | 2 | PLE task-specific experts (per task) |
| `EXPERT_DIM` | 128 | Expert FFN hidden size |
| `BATCH_SIZE` | 512 | Training batch size |
| `NUM_EPOCHS` | 50 | Max training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `PATIENCE` | 10 | Early-stopping patience |
| `W_CLICK` | 1.0 | Click loss weight |
| `W_RATING` | 0.5 | Rating loss weight |
| `MIN_INTERACTIONS` | 5 | 5-core filtering threshold |
| `NUM_NEG_EVAL` | 99 | Negative samples for evaluation |

---

## Implementation Notes

- **No external RecSys libraries** — all models implemented from scratch in PyTorch
- **Weight sharing**: item embeddings are shared between the encoder and the scoring layer (reduces parameters, improves generalisation)
- **Causal masking**: upper-triangular attention mask prevents attending to future items
- **Negative sampling**: one random negative per positive during training; 99 sampled negatives for evaluation (standard leave-one-out protocol)
- **Rating normalisation**: ratings normalised to [0, 1] for MSE stability; clamped to [1, 5] at inference
- **Early stopping**: on validation NDCG@10; best checkpoint used for test evaluation and ONNX export
- **Resume support**: all training scripts checkpoint optimizer and scheduler state and resume automatically if interrupted

---

## References

- Kang & McAuley (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). ICDM 2018.
- Ma et al. (2018). [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007). KDD 2018.
- Tang et al. (2020). [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236). RecSys 2020.
- Hou et al. (2024). [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952). Amazon Reviews 2023 dataset.
