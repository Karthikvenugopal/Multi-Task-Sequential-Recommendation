"""
Global configuration for the Multi-Task Sequential Recommendation system.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")

RAW_DATA_PATH = os.path.join(DATA_DIR, "Movies_and_TV.jsonl.gz")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA_URL = (
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
    "/raw/review_categories/Movies_and_TV.jsonl.gz"
)
MIN_INTERACTIONS = 5      # filter users/items below this count
MAX_SEQ_LEN = 50          # maximum sequence length fed to models
VAL_ITEMS = 1             # items held-out for validation (last-1)
TEST_ITEMS = 1            # items held-out for test (last)
NUM_NEG_EVAL = 99         # negative samples per positive during evaluation

# ── Model ──────────────────────────────────────────────────────────────────────
EMBED_DIM = 64            # item / position embedding size
NUM_HEADS = 2             # self-attention heads in SASRec blocks
NUM_BLOCKS = 2            # number of SASRec transformer blocks
DROPOUT = 0.2
NUM_EXPERTS = 4           # MMoE expert count
EXPERT_DIM = 128          # hidden dim of each expert FFN

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5              # early-stopping patience (epochs)

# MTL loss weights  (click_loss * w_click + rating_loss * w_rating)
W_CLICK = 1.0
W_RATING = 0.5

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "multitask_seqrec"
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR}"

# ── Serving ────────────────────────────────────────────────────────────────────
ONNX_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "mmoe_best.onnx")
BEST_MMOE_CKPT = os.path.join(CHECKPOINT_DIR, "mmoe_best.pt")
VOCAB_PATH = os.path.join(PROCESSED_DATA_PATH, "vocab.pkl")

# ── Device ─────────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
