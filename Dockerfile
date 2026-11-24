# Attack2Defense — single image for Render (API + React UI + CPU PyTorch).
# Tuned for Render Free (512 MB). If the process is killed on boot, RAM is the limit.

# --- frontend ---
FROM node:22-bookworm-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- runtime ---
FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels first so the image stays smaller than a CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY attack2defense ./attack2defense
COPY backend ./backend
RUN pip install --no-cache-dir -e ".[api]"

# Reported CSVs (Dashboard) + Small CNN weights (live Attack lab demos).
COPY deploy/publish/results ./results
COPY deploy/publish/checkpoints ./checkpoints

COPY --from=frontend /src/frontend/dist ./frontend/dist

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    A2D_DOWNLOAD_DATA=true \
    A2D_DATA_DIR=/app/data \
    A2D_CHECKPOINT_ROOTS='["/app/checkpoints"]' \
    A2D_RESULTS_ROOTS='["/app/results"]' \
    A2D_RUNS_DIR=/app/outputs/api_runs \
    A2D_MAX_TRAIN_EPOCHS=5 \
    A2D_MAX_TRAIN_BATCHES=80 \
    A2D_MAX_EVAL_IMAGES_PER_CLASS=20 \
    A2D_MAX_ATTACK_BATCH=4

EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}"]
