"""TrackingRunner - the live per-frame loop that generalizes deep_sort_app.run().

Two modes:
  * "live"  : detector -> conf-filter/NMS -> crop -> REID -> Detection -> tracker
  * "gtbox" : ground-truth boxes (via GtDetector) -> crop -> REID -> tracker
              (NMS/conf-filter skipped; SORT's detection step is effectively
               disabled so HOTA differences isolate the REID model)

Stage timings are recorded separately so FPS can be reported per component.
"""
import time
from dataclasses import dataclass

import numpy as np

from application_util import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from pipeline.crop import crop_patches


@dataclass
class RunStats:
    sequence: str
    n_frames: int
    det_time: float
    reid_time: float
    track_time: float
    total_time: float

    def _fps(self, t):
        return self.n_frames / t if t > 0 else 0.0

    @property
    def fps(self):
        return self._fps(self.total_time)

    def to_dict(self):
        return {
            "sequence": self.sequence,
            "n_frames": self.n_frames,
            "fps": round(self.fps, 3),
            "det_fps": round(self._fps(self.det_time), 3),
            "reid_fps": round(self._fps(self.reid_time), 3),
            "track_fps": round(self._fps(self.track_time), 3),
            "det_ms": round(1000 * self.det_time / max(self.n_frames, 1), 2),
            "reid_ms": round(1000 * self.reid_time / max(self.n_frames, 1), 2),
            "track_ms": round(1000 * self.track_time / max(self.n_frames, 1), 2),
        }


class TrackingRunner:
    def __init__(self, detector, reid, cfg, mode="live", body_reid=None):
        self.detector = detector
        self.reid = reid
        self.cfg = cfg
        self.mode = mode
        self.body_reid = body_reid

        tcfg = cfg.get("tracker", {})
        self.min_confidence = float(tcfg.get("min_confidence", 0.0))
        self.nms_max_overlap = float(tcfg.get("nms_max_overlap", 1.0))
        self.min_detection_height = int(tcfg.get("min_detection_height", 0))
        self.max_cosine_distance = float(tcfg.get("max_cosine_distance", 0.2))
        self.nn_budget = tcfg.get("nn_budget", None)
        self._tracker_kwargs = dict(
            max_iou_distance=float(tcfg.get("max_iou_distance", 0.7)),
            max_age=int(tcfg.get("max_age", 30)),
            n_init=int(tcfg.get("n_init", 3)),
        )
        self.crop_pad = float(cfg.get("reid", {}).get("crop_pad", 0.0))

    def _build_tracker(self):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        return Tracker(metric, **self._tracker_kwargs)

    def _filter(self, det):
        """Confidence-filter + NMS (skipped in gtbox mode)."""
        tlwh, conf = det.tlwh, det.confidence
        if self.mode == "gtbox":
            return tlwh, conf
        keep = conf >= self.min_confidence
        if self.min_detection_height > 0:
            keep = keep & (tlwh[:, 3] >= self.min_detection_height)
        tlwh, conf = tlwh[keep], conf[keep]
        if len(tlwh) > 0 and self.nms_max_overlap < 1.0:
            idxs = preprocessing.non_max_suppression(tlwh, self.nms_max_overlap, conf)
            tlwh, conf = tlwh[idxs], conf[idxs]
        return tlwh, conf

    def run_sequence(self, frame_source, writer=None, overlay=None):
        """Track an entire sequence. Returns RunStats.

        writer  : MotResultWriter | None     (results are written on completion)
        overlay : callable(frame_idx, frame, tracks, det_result, track_to_id) | None
        """
        tracker = self._build_tracker()
        # Warmup so reported FPS excludes one-time model-load / lazy-CUDA costs.
        try:
            self.detector.warmup(frame_source.image_size)
        except Exception:
            pass
        try:
            self.reid.warmup()
        except Exception:
            pass
        det_t = reid_t = trk_t = 0.0
        n = 0
        wall0 = time.perf_counter()

        for frame_idx, frame in frame_source:
            n += 1

            ts = time.perf_counter()
            det = self.detector.detect(frame, frame_idx)
            det_t += time.perf_counter() - ts

            tlwh, conf = self._filter(det)

            ts = time.perf_counter()
            patches = crop_patches(frame, tlwh, pad=self.crop_pad)
            feats = self.reid.extract(patches)
            reid_t += time.perf_counter() - ts

            detections = [Detection(tlwh[i], float(conf[i]), feats[i])
                          for i in range(len(tlwh))]

            ts = time.perf_counter()
            tracker.predict()
            tracker.update(detections)
            trk_t += time.perf_counter() - ts

            track_to_identity = None
            if self.body_reid is not None:
                track_to_identity = self.body_reid.process_frame(
                    frame_idx, frame, detections, tracker, feats,
                    fps=frame_source.fps)

            if writer is not None:
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    writer.append(frame_idx, track.track_id, track.to_tlwh())

            if overlay is not None:
                overlay(frame_idx, frame, tracker.tracks, det, track_to_identity)

        total = time.perf_counter() - wall0
        if writer is not None:
            writer.write()
        return RunStats(frame_source.name, n, det_t, reid_t, trk_t, total)
