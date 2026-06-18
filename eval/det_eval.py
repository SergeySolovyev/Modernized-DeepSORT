"""Detector evaluation vs ground truth: Precision / Recall / F1 @ IoU>=thr.

  python -m eval.det_eval --detector yolo --device cuda
  python -m eval.det_eval --detector nanodet --sequences TUD-Campus KITTI-17
"""
import argparse
import csv
import os

import numpy as np

from config import build_config
from detectors.registry import build_detector
from eval import journal
from eval.common import EVAL_SEQUENCES
from eval.iou import match_frame, prf1
from pipeline.frame_source import FrameSource


def _gt_detector(seq_dir, device):
    return build_detector({"name": "gt", "gt_file": os.path.join(seq_dir, "gt", "gt.txt"),
                           "pedestrian_classes": [1], "person_class_id": 1}, device)


def eval_on_sequence(detector_name, seq, data_root, device, iou_thr):
    cfg = build_config(detector=detector_name, sequence=seq)
    seq_dir = os.path.join(data_root, seq)
    fs = FrameSource(seq_dir)
    detector = build_detector(cfg, device)
    detector.warmup(fs.image_size)
    gt = _gt_detector(seq_dir, device)

    tp = fp = fn = 0
    for frame_idx, frame in fs:
        det = detector.detect(frame, frame_idx)
        ref = gt.detect(frame, frame_idx)
        a, b, c = match_frame(ref.tlwh, det.tlwh, iou_thr)
        tp += a
        fp += b
        fn += c
    p, r, f = prf1(tp, fp, fn)
    return {"detector": detector_name, "seq": seq, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def main():
    ap = argparse.ArgumentParser(description="Detector P/R/F1 vs GT")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for seq in args.sequences:
        try:
            row = eval_on_sequence(args.detector, seq, args.data_root, args.device, args.iou)
        except Exception as exc:  # keep going across sequences
            print("ERROR %s: %r" % (seq, exc))
            continue
        rows.append(row)
        print("%-16s P=%.3f R=%.3f F1=%.3f (tp=%d fp=%d fn=%d)"
              % (seq, row["precision"], row["recall"], row["f1"], row["tp"], row["fp"], row["fn"]))
        journal.append_row({"component": "detector", "iou": args.iou, **row})

    if rows:
        mp = float(np.mean([x["precision"] for x in rows]))
        mr = float(np.mean([x["recall"] for x in rows]))
        mf = float(np.mean([x["f1"] for x in rows]))
        print("MEAN             P=%.3f R=%.3f F1=%.3f" % (mp, mr, mf))
        out = args.out or os.path.join("results", "det_eval_%s.csv" % args.detector)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
