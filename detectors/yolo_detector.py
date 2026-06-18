"""Ultralytics YOLO detector (YOLOv8 / v11 family).

Source: https://github.com/ultralytics/ultralytics  (pip install ultralytics)
Ultralytics treats a numpy array input as BGR (cv2 convention), so we pass frames
through directly. Person filtering is done at inference via `classes=[person_id]`.
"""
import numpy as np

from .base import BaseDetector, DetectionResult, xyxy_to_tlwh
from .registry import register_detector


@register_detector("yolo")
class YoloDetector(BaseDetector):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        from ultralytics import YOLO

        self.model = YOLO(self.cfg.get("weights", "yolov8m.pt"))
        self.iou = float(self.cfg.get("iou", 0.7))
        self.imgsz = int(self.cfg.get("imgsz", 1280))
        self.half = bool(self.cfg.get("half", True)) and "cpu" not in str(device)

    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        results = self.model.predict(
            frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return DetectionResult.empty()
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(np.int64)
        return DetectionResult(xyxy_to_tlwh(xyxy), conf, cls)
