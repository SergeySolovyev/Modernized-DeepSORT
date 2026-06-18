"""Pluggable appearance (REID) feature extractors for the modernized DeepSORT.

Import `build_reid` from `reid.registry`. Concrete adapters are imported lazily by
the registry so a missing backend (e.g. fastreid) never blocks the others.
"""
from .base import BaseReIDExtractor  # noqa: F401
from .registry import build_reid, register_reid  # noqa: F401
