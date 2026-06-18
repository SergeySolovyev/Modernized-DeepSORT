"""Embed the GT body crops with a chosen REID model -> descriptors_<reid>.npz.

Input: data/gt_crops/manifest.csv (from data/prepare_gt_crops.py).
Output npz arrays: X (N,D float32, L2-normed), y_true (N global int labels),
seq (N str), frame (N int). Labels are global per (seq, track_id) because track ids
repeat across sequences.

  python -m bodyreid.eval.extract_gt --reid osnet_x1_0 --out descriptors_osnet.npz
"""
import argparse
import csv
import os

import cv2
import numpy as np

from config import build_config
from reid.registry import build_reid


def _load_manifest(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description="Embed GT crops for standalone REID eval")
    ap.add_argument("--reid", required=True, help="reid preset name (configs/reid/*.yaml)")
    ap.add_argument("--manifest", default=os.path.join("data", "gt_crops", "manifest.csv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    rows = _load_manifest(args.manifest)
    extractor = build_reid(build_config(reid=args.reid), args.device)

    label_of = {}
    feats, y_true, seqs, frames = [], [], [], []
    batch_imgs, batch_meta = [], []

    def flush():
        if not batch_imgs:
            return
        f = extractor.extract(batch_imgs)
        feats.append(np.asarray(f, dtype=np.float32))
        batch_imgs.clear()

    for r in rows:
        img = cv2.imread(r["crop_path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        key = (r["seq"], r["track_id"])
        label_of.setdefault(key, len(label_of))
        batch_imgs.append(img)
        y_true.append(label_of[key])
        seqs.append(r["seq"])
        frames.append(int(r["frame"]))
        if len(batch_imgs) >= args.batch:
            flush()
    flush()

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, 0), np.float32)
    out = args.out or "descriptors_%s.npz" % args.reid
    np.savez_compressed(out, X=X, y_true=np.asarray(y_true, np.int64),
                        seq=np.asarray(seqs), frame=np.asarray(frames, np.int64))
    print("wrote %s  X=%s  identities=%d" % (out, X.shape, len(label_of)))


if __name__ == "__main__":
    main()
