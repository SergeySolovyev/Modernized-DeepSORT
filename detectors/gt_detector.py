"""Ground-truth "detector" — returns GT boxes for the current frame.

Used for REID-only HOTA (SORT detection disabled: GT boxes in, vary only REID)
and as the reference for detector Precision/Recall/F1. Tolerant of both MOT16-style
gt (with consider-flag / class / visibility columns) and MOT15-style gt.
"""
import numpy as np

from .base import BaseDetector, DetectionResult
from .registry import register_detector


@register_detector("gt")
class GtDetector(BaseDetector):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        self.gt_file = self.cfg.get("gt_file")
        if not self.gt_file:
            raise ValueError("GtDetector requires cfg['gt_file'] (path to gt.txt)")
        self.min_visibility = float(self.cfg.get("min_visibility", 0.0))
        self.pedestrian_classes = self.cfg.get("pedestrian_classes", [1])
        self._by_frame = self._load(self.gt_file)

    def _load(self, path):
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data[None, :]
        ncol = data.shape[1]
        by_frame = {}
        for row in data:
            frame = int(row[0])
            x, y, w, h = row[2], row[3], row[4], row[5]
            flag = row[6] if ncol > 6 else 1.0          # MOT16 consider-flag / MOT15 conf
            # MOT16 gt is 9-col (class @7, visibility @8). MOT15 gt is 10-col where cols 7-9 are
            # 3D world coords (-1) — NOT class/visibility. Only read them when exactly 9 columns.
            cls = int(row[7]) if ncol == 9 else -1
            vis = row[8] if ncol == 9 else 1.0
            if flag == 0:                                # explicitly ignored GT
                continue
            if self.pedestrian_classes and cls != -1 and cls not in self.pedestrian_classes:
                continue
            if vis < self.min_visibility:
                continue
            by_frame.setdefault(frame, []).append((x, y, w, h))
        return by_frame

    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        if frame_idx is None:
            raise ValueError("GtDetector.detect needs frame_idx")
        boxes = self._by_frame.get(int(frame_idx), [])
        if not boxes:
            return DetectionResult.empty()
        tlwh = np.asarray(boxes, dtype=np.float32)
        conf = np.ones((len(tlwh),), dtype=np.float32)
        cls = np.full((len(tlwh),), self.person_class_id, dtype=np.int64)
        return DetectionResult(tlwh, conf, cls)
