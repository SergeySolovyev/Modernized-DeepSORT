"""Render an overlay MP4 (track boxes + ids, optional masks) for a config.

Produce the two overlays the rubric requires (original baseline + best config):
  python -m eval.make_overlays --detector gt   --reid mars       --sequence MOT16-09 --out overlays/baseline_MOT16-09.mp4
  python -m eval.make_overlays --detector yolo --reid osnet --sequence MOT16-09 --out overlays/best_MOT16-09.mp4
"""
import argparse
import os

import cv2
import numpy as np

from application_util.visualization import create_unique_color_uchar
from config import build_config
from detectors.registry import build_detector
from pipeline import FrameSource, TrackingRunner
from reid.registry import build_reid


class _LimitedSource:
    """Wrap a FrameSource to stop after n frames (for quick preview clips)."""
    def __init__(self, fs, n):
        self.fs, self.n = fs, n
        for attr in ("name", "fps", "image_size", "min_frame_idx", "max_frame_idx"):
            setattr(self, attr, getattr(fs, attr))

    def __iter__(self):
        for i, item in enumerate(self.fs):
            if self.n and i >= self.n:
                break
            yield item


class OverlayWriter:
    def __init__(self, out_path, fps):
        self.out_path = out_path
        self.fps = fps or 25
        self.writer = None
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    def __call__(self, frame_idx, frame, tracks, det, track_to_id=None):
        img = frame.copy()
        if det is not None and getattr(det, "masks", None) is not None and len(det.masks):
            ov = img.copy()
            for m in det.masks:
                ov[m] = (0, 200, 0)
            img = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            x, y, w, h = t.to_tlwh().astype(int)
            r, g, b = create_unique_color_uchar(t.track_id)
            color = (int(b), int(g), int(r))      # RGB -> BGR for cv2
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = "%d" % t.track_id
            if track_to_id and track_to_id.get(t.track_id) is not None:
                label += " #%s" % track_to_id[t.track_id]
            cv2.putText(img, label, (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.writer is None:
            hh, ww = img.shape[:2]
            self.writer = cv2.VideoWriter(self.out_path,
                                          cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (ww, hh))
        self.writer.write(img)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def _render_from_mot(seq_dir, mot_file, out_path):
    """Render an existing MOT-format result file (e.g. the baseline tracker) - no models."""
    data = np.atleast_2d(np.loadtxt(mot_file, delimiter=","))
    if data.size == 0 or data.shape[1] < 6:
        print("empty/invalid MOT file (nothing to render):", mot_file)
        return
    by_frame = {}
    for r in data:
        by_frame.setdefault(int(r[0]), []).append((int(r[1]), (r[2], r[3], r[4], r[5])))

    fs = FrameSource(seq_dir)
    writer = None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    for frame_idx, frame in fs:
        img = frame.copy()
        for tid, (x, y, w, h) in by_frame.get(frame_idx, []):
            r, g, b = create_unique_color_uchar(tid)
            color = (int(b), int(g), int(r))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(img, str(tid), (int(x), max(12, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if writer is None:
            hh, ww = img.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fs.fps or 25, (ww, hh))
        writer.write(img)
    if writer is not None:
        writer.release()
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser(description="Render tracking overlay MP4")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--reid", default="osnet")
    ap.add_argument("--sequence", default="MOT16-09")
    ap.add_argument("--mode", default="live", choices=["live", "gtbox"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--mot-file", default=None,
                    help="render this existing MOT result file (e.g. the baseline tracker) instead of running models")
    args = ap.parse_args()

    seq_dir = os.path.join(args.data_root, args.sequence)
    if args.mot_file:
        _render_from_mot(seq_dir, args.mot_file, args.out)
        return

    cfg = build_config(detector=args.detector, reid=args.reid, sequence=args.sequence)
    if args.mode == "gtbox":
        cfg["detector"] = {"name": "gt", "gt_file": os.path.join(seq_dir, "gt", "gt.txt"),
                           "pedestrian_classes": [1], "person_class_id": 1}
    elif cfg["detector"].get("name") == "gt" and not cfg["detector"].get("gt_file"):
        cfg["detector"]["gt_file"] = os.path.join(seq_dir, "gt", "gt.txt")

    fs = FrameSource(seq_dir)
    detector = build_detector(cfg, args.device)
    reid = build_reid(cfg, args.device)
    source = _LimitedSource(fs, args.max_frames) if args.max_frames else fs

    overlay = OverlayWriter(args.out, fs.fps)
    runner = TrackingRunner(detector, reid, cfg, mode=args.mode)
    runner.run_sequence(source, overlay=overlay)
    overlay.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
