#!/usr/bin/env bash
# Rebuild and restart Attack2Defense on a single EC2 host without losing accounts.
# Usage (on the server):  bash scripts/ec2_redeploy.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PERSIST="${A2D_PERSIST_DIR:-$HOME/a2d-persist}"
IMAGE="${A2D_IMAGE_NAME:-attack2defense}"
NAME="${A2D_CONTAINER_NAME:-a2d}"

mkdir -p "$PERSIST/data" "$PERSIST/db" "$PERSIST/runs"
if [[ ! -f "$PERSIST/secret.txt" ]]; then
  openssl rand -hex 32 > "$PERSIST/secret.txt"
  chmod 600 "$PERSIST/secret.txt"
  echo "Created new JWT secret at $PERSIST/secret.txt (kept across rebuilds)."
fi
SECRET="$(tr -d '\n\r' < "$PERSIST/secret.txt")"

cd "$ROOT"
git pull --ff-only
docker build -t "$IMAGE" .
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --restart unless-stopped \
  -p 80:8000 \
  -v "$PERSIST/data:/app/data" \
  -v "$PERSIST/db:/app/persist" \
  -v "$PERSIST/runs:/app/outputs/api_runs" \
  -e A2D_SECRET_KEY="$SECRET" \
  -e A2D_DATABASE_URL="sqlite:////app/persist/attack2defense.db" \
  -e A2D_DATA_DIR=/app/data \
  -e A2D_RUNS_DIR=/app/outputs/api_runs \
  -e A2D_DOWNLOAD_DATA=true \
  -e A2D_MAX_TRAIN_EPOCHS=5 \
  -e A2D_MAX_TRAIN_BATCHES=80 \
  -e A2D_MAX_EVAL_IMAGES_PER_CLASS=20 \
  -e A2D_MAX_ATTACK_BATCH=4 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  "$IMAGE"

echo "Container $NAME is up. Accounts/CIFAR persist under $PERSIST"
docker ps --filter "name=^/${NAME}$"
docker logs --tail 15 "$NAME"
