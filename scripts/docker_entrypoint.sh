#!/bin/sh
# Ensure CIFAR-10 exists (persisted volume or first boot), then start the API.
set -eu
python - <<'PY'
from pathlib import Path

from torchvision.datasets import CIFAR10

root = Path("/app/data")
root.mkdir(parents=True, exist_ok=True)
ready = (root / "cifar-10-batches-py" / "test_batch").exists()
if not ready:
    print("Downloading CIFAR-10 into", root, flush=True)
    CIFAR10(str(root), train=True, download=True)
    CIFAR10(str(root), train=False, download=True)
    print("CIFAR-10 ready", flush=True)
else:
    print("CIFAR-10 already present", flush=True)
PY
exec uvicorn backend.app.main:create_app --factory --host 0.0.0.0 --port "${PORT:-8000}"
