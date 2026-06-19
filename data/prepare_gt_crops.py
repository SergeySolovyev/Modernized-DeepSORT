"""Export ground-truth body crops labeled by true track id.

These crops feed the STANDALONE REID evaluation (bodyreid/eval), which clusters
descriptors and scores them with Fowlkes-Mallows / Silhouette / Calinski-Harabasz.

  python -m data.prepare_gt_crops --sequences TUD-Campus MOT16-09 --out data/gt_crops

Output:
  <out>/<seq>/<track_id>/<frame>.jpg
  <out>/manifest.csv  (columns: crop_path, seq, track_id, frame)
"""
import argparse
import csv
import os

import cv2
import numpy as np

from data.seqinfo import list_frames, read_seqinfo

DEFAULT_SEQS = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1",
                "MOT16-09", "MOT16-11"]


def _parse_gt(path, pedestrian_classes=(1,), min_visibility=0.0):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[None, :]
    ncol = data.shape[1]
    by_frame = {}
    for row in data:
        frame, tid = int(row[0]), int(row[1])
        x, y, w, h = row[2], row[3], row[4], row[5]
        flag = row[6] if ncol > 6 else 1.0
        # MOT16 gt is 9-col (..,class,visibility). MOT15/2DMOT2015 gt is 10-col where cols
        # 7-9 are 3D world coords (-1 for 2D seqs) - NOT class/visibility. Only trust them at 9 cols.
        cls = int(row[7]) if ncol == 9 else -1
        vis = row[8] if ncol == 9 else 1.0
        if flag == 0:
            continue
        if pedestrian_classes and cls != -1 and cls not in pedestrian_classes:
            continue
        if vis < min_visibility:
            continue
        by_frame.setdefault(frame, []).append((tid, x, y, w, h))
    return by_frame


def export_sequence(seq, data_root, out_root, manifest_rows):
    seq_dir = os.path.join(data_root, seq)
    gt_path = os.path.join(seq_dir, "gt", "gt.txt")
    if not os.path.exists(gt_path):
        print("  skip %s (no gt.txt)" % seq)
        return 0
    by_frame = _parse_gt(gt_path)
    frames = dict(list_frames(seq_dir, read_seqinfo(seq_dir)))
    n = 0
    for frame_idx, boxes in by_frame.items():
        path = frames.get(frame_idx)
        if path is None:
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        for tid, x, y, w, h in boxes:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            dst_dir = os.path.join(out_root, seq, str(tid))
            os.makedirs(dst_dir, exist_ok=True)
            crop_path = os.path.join(dst_dir, "%06d.jpg" % frame_idx)
            cv2.imwrite(crop_path, crop)
            manifest_rows.append({"crop_path": crop_path, "seq": seq,
                                  "track_id": tid, "frame": frame_idx})
            n += 1
    print("  %s: %d crops" % (seq, n))
    return n


def main():
    ap = argparse.ArgumentParser(description="Export GT body crops for standalone REID eval")
    ap.add_argument("--sequences", nargs="*", default=DEFAULT_SEQS)
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--out", default="data/gt_crops")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_rows = []
    for seq in args.sequences:
        export_sequence(seq, args.data_root, args.out, manifest_rows)

    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop_path", "seq", "track_id", "frame"])
        w.writeheader()
        w.writerows(manifest_rows)
    print("manifest -> %s (%d crops)" % (manifest, len(manifest_rows)))


if __name__ == "__main__":
    main()
