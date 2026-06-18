"""REID registry + factory with lazy backend imports."""
import importlib

_REGISTRY = {}

# name -> module that defines & registers the adapter (imported on demand).
_MODULE_BY_NAME = {
    "torchreid": "reid.torchreid_extractor",
    "boxmot": "reid.boxmot_extractor",
    "timm": "reid.timm_extractor",
    "fastreid": "reid.fastreid_extractor",
    "mars": "reid.mars_extractor",
}


def register_reid(name):
    def _decorator(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _decorator


def available_reid():
    return sorted(set(list(_REGISTRY) + list(_MODULE_BY_NAME)))


def build_reid(cfg, device="cuda"):
    """Instantiate a REID extractor from the merged `cfg["reid"]` dict.

    `cfg` may be the full run config (with a "reid" key) or the reid sub-dict
    directly. The backend is keyed by `name` (e.g. "torchreid"); the specific
    model within a backend is selected by backend-specific keys (e.g.
    `model_name: osnet_x1_0`).
    """
    reid_cfg = cfg.get("reid", cfg)
    name = reid_cfg.get("name")
    if name is None:
        raise ValueError("reid config missing 'name'")
    if name not in _REGISTRY:
        module = _MODULE_BY_NAME.get(name)
        if module is None:
            raise KeyError("Unknown reid backend '%s'. Available: %s"
                           % (name, available_reid()))
        importlib.import_module(module)
    if name not in _REGISTRY:
        raise KeyError("REID module for '%s' did not register itself" % name)
    return _REGISTRY[name](reid_cfg, device)
