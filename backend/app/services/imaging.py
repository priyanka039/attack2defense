"""Convert between image tensors and base64 PNG data URLs."""

from __future__ import annotations

import base64
import binascii
import io

import matplotlib
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.app.errors import ApiError

_PREFIX = "data:image/png;base64,"
_HEATMAP = matplotlib.colormaps["inferno"]


def _encode(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return _PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def to_data_url(image: torch.Tensor) -> str:
    """Encode a CHW tensor in [0, 1] as a PNG data URL."""

    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return _encode(array)


def perturbation_url(delta: torch.Tensor, epsilon: float) -> str:
    """Show the signed perturbation with gray as zero and full contrast at +/- epsilon."""

    scale = max(epsilon, 1e-12)
    return to_data_url((delta / (2 * scale) + 0.5).clamp(0, 1))


def heatmap_url(delta: torch.Tensor, epsilon: float) -> str:
    """Per-pixel perturbation size (largest channel change) as a colour heatmap."""

    magnitude = (delta.detach().abs().amax(dim=0) / max(epsilon, 1e-12)).clamp(0, 1).cpu().numpy()
    colours = (_HEATMAP(magnitude)[..., :3] * 255).astype(np.uint8)
    return _encode(colours)


def from_data_url(data: str, size: int) -> torch.Tensor:
    """Decode an uploaded PNG or JPEG (data URL or bare base64) into a CHW tensor."""

    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(payload, validate=True)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as exc:
        raise ApiError(
            422, "invalid_image", "The uploaded file is not a readable PNG or JPEG image.", "image"
        ) from exc
    image = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
