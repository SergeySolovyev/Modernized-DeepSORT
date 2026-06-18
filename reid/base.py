"""Appearance (REID) extractor contract.

An extractor maps a list of person crops (BGR uint8 HxWx3, variable size) to an
(N, D) array of L2-normalized float32 embeddings. The feature dimensionality D is
model-specific (e.g. 512 for OSNet, 2048 for ResNet50, 128 for mars-small128); the
DeepSORT cosine metric is dimension-agnostic, so the rest of the pipeline does not
care about D as long as it is consistent within a run.
"""
from abc import ABC, abstractmethod

import numpy as np


def l2_normalize(feats, eps=1e-12):
    """Row-wise L2 normalization of an (N, D) array."""
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats[None, :]
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    return feats / np.maximum(norms, eps)


class BaseReIDExtractor(ABC):
    """Abstract appearance feature extractor.

    Subclasses receive the merged `cfg["reid"]` sub-dict and a device string.
    """

    name = "base"

    def __init__(self, cfg, device="cuda"):
        self.cfg = dict(cfg or {})
        self.device = device
        self._feature_dim = None

    @abstractmethod
    def _extract_raw(self, patches) -> np.ndarray:
        """Return (N, D) raw (un-normalized) features for a list of BGR crops."""
        raise NotImplementedError

    def extract(self, patches) -> np.ndarray:
        """Return (N, D) L2-normalized float32 embeddings for BGR crops.

        Empty input returns a correctly-shaped (0, D) array when D is known,
        else (0, 0).
        """
        if patches is None or len(patches) == 0:
            dim = self._feature_dim or 0
            return np.zeros((0, dim), dtype=np.float32)
        raw = np.asarray(self._extract_raw(patches), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw[None, :]
        if self._feature_dim is None:
            self._feature_dim = int(raw.shape[1])
        return l2_normalize(raw)

    @property
    def feature_dim(self):
        """Embedding dimensionality (probed lazily on first extract if unknown)."""
        return self._feature_dim

    def warmup(self):
        """Run one dummy crop so FPS timing excludes lazy init costs."""
        try:
            self.extract([np.zeros((128, 64, 3), dtype=np.uint8)])
        except Exception:
            pass
