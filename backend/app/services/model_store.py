"""Look up models from the registry and keep recently used ones in memory."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from attack2defense.evaluate import restore_model
from attack2defense.registry import ModelEntry, discover_models
from attack2defense.utils import get_device
from backend.app.errors import ApiError, not_found
from backend.app.settings import Settings

_REFRESH_SECONDS = 10.0
_CACHE_SIZE = 4


class ModelStore:
    """Public models come from the configured roots; each user also sees their own runs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = get_device()
        self._lock = threading.Lock()
        self._public: dict[str, ModelEntry] = {}
        self._public_loaded_at = 0.0
        self._loaded: OrderedDict[tuple[str, float], torch.nn.Module] = OrderedDict()

    def user_dir(self, user_id: int) -> Path:
        return Path(self.settings.runs_dir) / str(user_id)

    def public_entries(self) -> dict[str, ModelEntry]:
        with self._lock:
            if time.monotonic() - self._public_loaded_at > _REFRESH_SECONDS:
                self._public = discover_models(self.settings.checkpoint_roots, self.settings.results_roots)
                self._public_loaded_at = time.monotonic()
            return dict(self._public)

    def user_entries(self, user_id: int) -> dict[str, ModelEntry]:
        folder = self.user_dir(user_id)
        return discover_models([folder], [folder])

    def entries(self, user_id: int | None) -> dict[str, ModelEntry]:
        entries = self.public_entries()
        if user_id is not None:
            entries.update(self.user_entries(user_id))
        return entries

    def get(self, model_id: str, user_id: int | None) -> ModelEntry:
        entry = self.entries(user_id).get(model_id)
        if entry is None:
            raise not_found("model", model_id)
        return entry

    def load(self, entry: ModelEntry) -> torch.nn.Module:
        if entry.checkpoint is None:
            raise ApiError(
                409,
                "model_has_no_weights",
                f"Results for '{entry.id}' exist, but its best.pt is missing. "
                f"Download it into checkpoints/{entry.id}/ to attack or evaluate it.",
                "model_id",
            )
        key = (str(entry.checkpoint), entry.checkpoint.stat().st_mtime)
        with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._loaded[key]
        model, _, _ = restore_model(entry.checkpoint, self.device)
        with self._lock:
            self._loaded[key] = model
            while len(self._loaded) > _CACHE_SIZE:
                self._loaded.popitem(last=False)
        return model

    def invalidate(self) -> None:
        with self._lock:
            self._public_loaded_at = 0.0
