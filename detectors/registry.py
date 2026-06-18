"""Detector registry + factory with lazy backend imports.

Each adapter module registers its class via @register_detector("name"). The
factory imports the adapter module only when that detector is requested, so
heavy optional deps (ultralytics, mmdet, nanodet) are never imported unless used.
"""
import importlib

_REGISTRY = {}

# name -> module that defines & registers the adapter (imported on demand).
_MODULE_BY_NAME = {
    "yolo": "detectors.yolo_detector",
    "yolo_seg": "detectors.yolo_seg_detector",
    "nanodet": "detectors.nanodet_detector",
    "mmdet": "detectors.mmdet_detector",
    "gt": "detectors.gt_detector",
}


def register_detector(name):
    def _decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _decorator


def available_detectors():
    return sorted(set(list(_REGISTRY) + list(_MODULE_BY_NAME)))


def build_detector(cfg, device="cuda"):
    """Instantiate a detector from the merged `cfg["detector"]` dict.

    `cfg` may be the full run config (with a "detector" key) or the detector
    sub-dict directly. The chosen backend is keyed by `name`.
    """
    det_cfg = cfg.get("detector", cfg)
    name = det_cfg.get("name")
    if name is None:
        raise ValueError("detector config missing 'name'")
    if name not in _REGISTRY:
        module = _MODULE_BY_NAME.get(name)
        if module is None:
            raise KeyError("Unknown detector '%s'. Available: %s"
                           % (name, available_detectors()))
        importlib.import_module(module)
    if name not in _REGISTRY:
        raise KeyError("Detector module for '%s' did not register itself" % name)
    return _REGISTRY[name](det_cfg, device)
