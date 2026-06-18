"""Ultralytics YOLOv8-seg instance-segmentation detector (segmentation bonus).

Produces person masks; the tracking bbox is derived from the tight mask extent
when `mask_to_bbox` is set, otherwise the detector's own box is used. Full-res
boolean masks are returned in DetectionResult.masks for the segmentation overlay.
"""
import cv2
import numpy as np

from .base import BaseDetector, DetectionResult, xyxy_to_tlwh
from .registry import register_detector


@register_detector("yolo_seg")
class YoloSegDetector(BaseDetector):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        from ultralytics import YOLO

        self.model = YOLO(self.cfg.get("weights", "yolov8m-seg.pt"))
        self.iou = float(self.cfg.get("iou", 0.7))
        self.imgsz = int(self.cfg.get("imgsz", 1280))
        self.half = bool(self.cfg.get("half", True)) and "cpu" not in str(device)
        self.mask_to_bbox = bool(self.cfg.get("mask_to_bbox", True))

    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            classes=[self.person_class_id], device=self.device,
            half=self.half, verbose=False,
            retina_masks=True,   # masks at original frame HxW (not letterboxed square)
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return DetectionResult.empty()

        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(np.int64)
        xyxy = r.boxes.xyxy.cpu().numpy()

        masks_full = None
        if r.masks is not None:
            # With retina_masks=True, r.masks.data is binary uint8 (0/1) at the
            # original frame HxW; resize is then a safe no-op (kept for robustness).
            md = r.masks.data.cpu().numpy()
            masks_full = np.zeros((md.shape[0], h, w), dtype=bool)
            for i in range(md.shape[0]):
                m = cv2.resize(md[i], (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
                masks_full[i] = m
            if self.mask_to_bbox:
                xyxy = xyxy.copy()
                for i in range(masks_full.shape[0]):
                    ys, xs = np.where(masks_full[i])
                    if xs.size and ys.size:
                        xyxy[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

        return DetectionResult(xyxy_to_tlwh(xyxy), conf, cls, masks=masks_full)
