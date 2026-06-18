"""Detection backend contract.

Every detector normalizes its native output to a `DetectionResult` carrying
person-only boxes in MOT/DeepSORT `tlwh` format, so the rest of the pipeline
(`application_util.preprocessing.non_max_suppression`, `deep_sort.detection.Detection`)
consumes them unchanged regardless of the underlying model.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


def xyxy_to_tlwh(xyxy):
    """Convert (...,4) [x1,y1,x2,y2] boxes to [x,y,w,h] (top-left + size)."""
    xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    tlwh = xyxy.copy()
    tlwh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    tlwh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return tlwh


@dataclass
class DetectionResult:
    """Per-frame detector output, filtered to the person class.

    Attributes
    ----------
    tlwh : (N, 4) float32      bounding boxes [x, y, w, h]
    confidence : (N,) float32  detector scores
    class_ids : (N,) int64     class index per box (person id, kept for clarity)
    masks : Optional[(N, H, W) bool]  instance masks (segmentation backends only)
    """
    tlwh: np.ndarray
    confidence: np.ndarray
    class_ids: np.ndarray
    masks: Optional[np.ndarray] = None

    def __post_init__(self):
        self.tlwh = np.asarray(self.tlwh, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)

    def __len__(self):
        return int(self.tlwh.shape[0])

    @staticmethod
    def empty():
        return DetectionResult(
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
            None,
        )


class BaseDetector(ABC):
    """Abstract person detector.

    Subclasses implement `detect(frame_bgr) -> DetectionResult` returning
    person-only boxes. The constructor receives the merged `cfg["detector"]`
    sub-dict and a device string.
    """

    name = "base"

    def __init__(self, cfg, device="cuda"):
        self.cfg = dict(cfg or {})
        self.device = device
        # `person_class_id` is centralized here — never hardcode 0 in a subclass,
        # because the COCO person index can differ across frameworks/configs.
        self.person_class_id = int(self.cfg.get("person_class_id", 0))
        self.conf = float(self.cfg.get("conf", 0.25))

    @abstractmethod
    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        """Run inference on a single BGR uint8 HxWx3 frame.

        `frame_idx` is passed by the runner and used only by `GtDetector`
        (to look up ground-truth boxes); real detectors ignore it.
        """
        raise NotImplementedError

    def warmup(self, image_size=(720, 1280)):
        """Run one dummy forward so FPS timing excludes lazy CUDA/init costs."""
        try:
            dummy = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            self.detect(dummy)
        except Exception:  # warmup is best-effort; never fatal
            pass
