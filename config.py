"""Config loading + merging for the modernized DeepSORT.

Model selection happens *before* execution via named YAML presets:
  configs/default.yaml           global defaults (tracker params, paths, device)
  configs/detectors/<name>.yaml  a detector preset (picks backend + weights + params)
  configs/reid/<name>.yaml       a REID preset (picks backend + model + params)
  configs/sequences/<seq>.yaml   optional per-VIDEO parameter overrides

Merge order (low -> high precedence):
  default -> detector preset -> reid preset -> sequence preset -> CLI --override

The merged dict is consumed by build_detector / build_reid and the Tracker.
"""
import copy
import os

import yaml

# Repo root = directory containing this file.
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "configs")


def _deep_merge(base, override):
    """Recursively merge `override` into `base` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(path):
    """Load a YAML file, returning {} if it is missing or empty."""
    if path is None or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _coerce(value):
    """Best-effort scalar coercion for CLI --override values."""
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg, overrides):
    """Apply a list of dotted 'a.b.c=value' strings onto cfg (in place)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("Override '%s' must be key=value" % item)
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = _coerce(raw.strip())
    return cfg


def build_config(detector=None, reid=None, sequence=None, overrides=None,
                 config_dir=CONFIG_DIR):
    """Assemble the full run config from named presets + CLI overrides."""
    cfg = load_yaml(os.path.join(config_dir, "default.yaml"))
    if detector is not None:
        preset = load_yaml(os.path.join(config_dir, "detectors", detector + ".yaml"))
        if not preset:
            raise FileNotFoundError("No detector preset 'configs/detectors/%s.yaml'" % detector)
        cfg = _deep_merge(cfg, preset)
    if reid is not None:
        preset = load_yaml(os.path.join(config_dir, "reid", reid + ".yaml"))
        if not preset:
            raise FileNotFoundError("No reid preset 'configs/reid/%s.yaml'" % reid)
        cfg = _deep_merge(cfg, preset)
    if sequence is not None:
        # Sequence presets are optional - silently skip if absent.
        cfg = _deep_merge(cfg, load_yaml(os.path.join(config_dir, "sequences", sequence + ".yaml")))
    cfg = apply_overrides(cfg, overrides)
    return cfg
