"""Tunable parameters for the standalone body-REID identity system.

Defaults are sensible starting points; bodyreid/eval/sweep.py optimizes them on
GT crops. Every threshold here is exactly what the standalone clustering eval tunes.
"""
from dataclasses import dataclass


@dataclass
class ReidConfig:
    # --- create vs match (cosine distance d = 1 - cos_sim on L2-normed vectors) ---
    match_thresh: float = 0.30      # d <= this -> assign + enroll into gallery
    new_thresh: float = 0.45        # d >  this -> create a new identity
    #   match_thresh < d <= new_thresh -> "ambiguous band": assign tentatively, do NOT enroll
    merge_thresh: float = 0.10      # offline merge of identities with centroid distance below this

    # --- nearest-neighbour search policy ---
    k: int = 5
    search_target: str = "centroid"      # "centroid" | "per_descriptor"
    search_rule: str = "k1"              # "k1" | "knn_vote" | "radius"

    # --- gallery management (per identity) ---
    gallery_cap: int = 50
    gallery_evict: str = "quality"       # "fifo" | "quality" | "diversity"
    gallery_stride: int = 5              # admit at most 1 / stride descriptors per track

    # --- time-window identity resolution per track ---
    T_seconds: float = 2.0
    vote_weight: str = "distance"        # "none" | "distance"
    vote_min_conf: float = 0.55          # hysteresis: keep previous iid if winner conf below this

    # --- cross-track conflict resolution ---
    conflict_policy: str = "keep_best"   # "reset_all" | "keep_best" | "keep_best_spawn"

    # --- housekeeping ---
    identity_ttl_seconds: float = 1800.0
    index_rebuild_every: int = 500
    iou_assoc_thresh: float = 0.5        # detection -> track association IoU

    # --- crop quality gate (skip tiny/low-conf crops when enrolling) ---
    quality_min_area: float = 48 * 96
    quality_min_conf: float = 0.5
