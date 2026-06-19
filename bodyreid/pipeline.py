"""BodyReidRunner - the per-frame hook called by pipeline.runner after tracker.update.

Maintains a persistent identity database (iid space) decoupled from the tracker's
ephemeral track ids (tid space), healing fragmentation: when the tracker splits one
person into tid=5 then tid=9, both resolve to the same iid via gallery match.

Pipeline per frame (matches the assignment spec):
  1. descriptors (reuse the tracker's REID embeddings, or a 2nd model)
  2. tracking step already done -> associate each detection to a track by IoU
  3. kNN-search the identity DB per descriptor -> known iid or create new (on track birth)
  4. append (timestamp, iid, dist) to the track's identity history
  5. resolve every active track over a time window T (weighted majority vote)
  6. resolve cross-track identity conflicts
  7. TTL eviction
"""
import numpy as np

from eval.iou import iou_matrix

from .conflicts import ConflictManager
from .config import ReidConfig
from .identity_db import IdentityDatabase
from .resolver import IdentityResolver


class BodyReidRunner:
    def __init__(self, cfg=None, reid=None):
        self.cfg = cfg or ReidConfig()
        self.reid = reid                 # optional dedicated 2nd REID model
        self.db = IdentityDatabase(self.cfg)
        self.resolver = IdentityResolver(self.cfg)
        self.conflicts = ConflictManager(self.cfg)
        self.seen_tracks = set()
        self._frame = 0

    def _associate(self, detections, confirmed):
        """detection index -> track_id via max-IoU >= iou_assoc_thresh."""
        if not detections or not confirmed:
            return {}
        det_boxes = np.asarray([d.tlwh for d in detections], dtype=np.float32)
        trk_boxes = np.asarray([t.to_tlwh() for t in confirmed], dtype=np.float32)
        m = iou_matrix(det_boxes, trk_boxes)          # (Nd, Nt)
        out = {}
        for i in range(len(detections)):
            j = int(np.argmax(m[i]))
            if m[i, j] >= self.cfg.iou_assoc_thresh:
                out[i] = confirmed[j].track_id
        return out

    def _quality(self, det):
        area = float(det.tlwh[2]) * float(det.tlwh[3])
        good = area >= self.cfg.quality_min_area and det.confidence >= self.cfg.quality_min_conf
        return 1.0 if good else 0.3

    def process_frame(self, frame_idx, frame, detections, tracker, feats, fps=30):
        fps = fps or 30
        ts = frame_idx / float(fps)

        if self.reid is not None and len(detections):
            from pipeline.crop import crop_patches
            feats = self.reid.extract(crop_patches(frame, [d.tlwh for d in detections]))

        confirmed = [t for t in tracker.tracks if t.is_confirmed()]
        active_tids = {t.track_id for t in confirmed}
        det_tid = self._associate(detections, confirmed)

        for i, det in enumerate(detections):
            tid = det_tid.get(i)
            if tid is None:
                continue
            vec = np.asarray(feats[i], dtype=np.float32)
            quality = self._quality(det)
            iid, dist, status = self.db.search(vec)
            if status == "miss" or iid is None:
                if tid not in self.seen_tracks:           # create only on track birth
                    iid = self.db.create(vec, ts, tid)
                    dist = 0.0
                else:
                    iid = None
            elif quality >= 0.999:                        # enroll only good crops
                self.db.maybe_enroll(iid, vec, ts, tid, quality, status)
            self.resolver.record(tid, ts, iid, dist if iid is not None else 1.0)
            self.seen_tracks.add(tid)

        resolved = {tid: self.resolver.resolve(tid, ts, fps) for tid in active_tids}
        resolved = self.conflicts.resolve(resolved, self.resolver.last_strength, self.resolver)

        self._frame += 1
        if self._frame % max(self.cfg.index_rebuild_every, 1) == 0:
            self.db.evict_ttl(ts)
        return resolved
