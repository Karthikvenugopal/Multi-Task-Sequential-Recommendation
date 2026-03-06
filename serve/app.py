"""
FastAPI recommendation serving endpoint.

Endpoints:
    POST /recommend     — return top-K item recommendations
    POST /score         — score specific candidate items
    GET  /health        — liveness probe
    GET  /info          — model/catalogue info

Usage:
    uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ONNX_MODEL_PATH, VOCAB_PATH
from serve.onnx_inference import OnnxRecommender

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("seqrec.serve")

# ── Global recommender instance (loaded at startup) ────────────────────────────
_recommender: Optional[OnnxRecommender] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _recommender
    logger.info("Loading ONNX model and vocabulary …")
    t0 = time.time()
    try:
        _recommender = OnnxRecommender(
            onnx_path=ONNX_MODEL_PATH,
            vocab_path=VOCAB_PATH,
        )
        logger.info(
            f"Model loaded in {time.time() - t0:.2f}s  "
            f"({_recommender.num_items:,} items in catalogue)"
        )
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
        # We let the app start even if model fails — /health will report it
    yield
    # Shutdown — nothing to clean up for ONNX Runtime


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Task Sequential Recommender",
    description=(
        "SASRec + MMoE recommendation API. "
        "Accepts a user interaction history and returns top-K items."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    """
    Request body for the /recommend endpoint.

    item_ids: chronological list of item identifiers the user has interacted with.
              Item IDs are the raw ASIN strings from the Amazon catalogue.
    top_k:    number of recommendations to return (1–100).
    exclude_seen: if True (default), already-interacted items are excluded.
    """
    item_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Chronological interaction history (list of item IDs / ASINs)",
        examples=[["B001E4KFG0", "B00813GRG4", "B00KQPKJK4"]],
    )
    top_k: int = Field(
        default=10, ge=1, le=100,
        description="Number of recommendations to return"
    )
    exclude_seen: bool = Field(
        default=True,
        description="Exclude already-interacted items from recommendations"
    )

    @field_validator("item_ids")
    @classmethod
    def strip_whitespace(cls, v):
        return [x.strip() for x in v if x.strip()]


class RecommendedItem(BaseModel):
    rank: int
    item_id: str
    internal_id: int
    score: float


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedItem]
    num_results: int
    latency_ms: float


class ScoreRequest(BaseModel):
    """Score a specific list of candidate items."""
    item_ids: List[str] = Field(..., min_length=1, max_length=500)
    candidate_ids: List[str] = Field(..., min_length=1, max_length=1000)


class ScoreResponse(BaseModel):
    scores: Dict[str, float]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_items: Optional[int] = None


class InfoResponse(BaseModel):
    model: str
    num_items: int
    max_seq_len: int
    onnx_path: str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _get_recommender() -> OnnxRecommender:
    if _recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender model is not loaded. Check server logs."
        )
    return _recommender


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health() -> HealthResponse:
    """Liveness probe."""
    loaded = _recommender is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        num_items=_recommender.num_items if loaded else None,
    )


@app.get("/info", response_model=InfoResponse, tags=["Meta"])
def info() -> InfoResponse:
    """Return model and catalogue metadata."""
    rec = _get_recommender()
    from config import MAX_SEQ_LEN
    return InfoResponse(
        model="SASRec + MMoE (multi-task)",
        num_items=rec.num_items,
        max_seq_len=MAX_SEQ_LEN,
        onnx_path=ONNX_MODEL_PATH,
    )


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendation"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Return top-K recommended items for a user interaction sequence.

    The model is a SASRec + MMoE multi-task model trained jointly on
    CTR prediction (click) and rating regression.

    Scores reflect predicted click likelihood (higher = more relevant).
    """
    rec = _get_recommender()

    t0 = time.perf_counter()
    try:
        results = rec.recommend(
            item_ids=request.item_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"/recommend  seq_len={len(request.item_ids)}  "
        f"top_k={request.top_k}  {elapsed_ms:.1f}ms"
    )
    return RecommendResponse(
        recommendations=[RecommendedItem(**r) for r in results],
        num_results=len(results),
        latency_ms=round(elapsed_ms, 2),
    )


@app.post("/score", response_model=ScoreResponse, tags=["Recommendation"])
def score_candidates(request: ScoreRequest) -> ScoreResponse:
    """
    Score a specific set of candidate items given a user interaction sequence.

    Useful for re-ranking externally retrieved candidates.
    """
    rec = _get_recommender()

    t0 = time.perf_counter()
    try:
        scores = rec.score_candidates(
            item_ids=request.item_ids,
            candidate_ids=request.candidate_ids,
        )
    except Exception as exc:
        logger.exception("Scoring error")
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        scores={cid: float(s) for cid, s in zip(request.candidate_ids, scores)},
        latency_ms=round(elapsed_ms, 2),
    )
