# ──────────────────────────────────────────────────────────────────────────────
# Multi-Task Sequential Recommendation — Docker image
#
# Build:  docker build -t seqrec:latest .
# Run:    docker run -p 8000:8000 \
#             -v $(pwd)/checkpoints:/app/checkpoints \
#             -v $(pwd)/data/processed:/app/data/processed \
#             seqrec:latest
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# ── OS dependencies ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# Install CPU-only PyTorch first (avoids pulling 3 GB CUDA wheels in prod)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ───────────────────────────────────────────────────────────
COPY config.py          ./config.py
COPY models/            ./models/
COPY serve/             ./serve/
COPY data/dataset.py    ./data/dataset.py

# ── Runtime environment ────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serve.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
