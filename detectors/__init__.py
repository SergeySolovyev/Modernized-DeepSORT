"""Pluggable person-detection backends for the modernized DeepSORT.

Import `build_detector` from `detectors.registry`. Concrete adapters are imported
lazily by the registry so that one heavy/broken backend (e.g. mmdet) never blocks
the others at import time.
"""
from .base import BaseDetector, DetectionResult, xyxy_to_tlwh  # noqa: F401
from .registry import build_detector, register_detector  # noqa: F401
