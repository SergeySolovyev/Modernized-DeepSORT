"""End-to-end pipeline test on a synthetic MOT sequence (no GPU / no models).

Builds a tiny sequence with two distinctly-colored people, runs the full
TrackingRunner in gtbox mode with a stub REID extractor (feature = mean BGR color,
so each identity is appearance-stable), and checks that MOT results come out with
the expected number of identities. This exercises FrameSource, GtDetector,
crop_patches, the REID -> Detection path, the Tracker, and MotResultWriter together.

Run: py -3.14 tests/test_pipeline_e2e.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid.base import BaseReIDExtractor  # noqa: E402


class StubReID(BaseReIDExtractor):
    """Appearance = mean BGR color of the crop (stable per identity)."""
    def _extract_raw(self, patches):
        return np.asarray([p.reshape(-1, 3).mean(axis=0) + 1.0 for p in patches],
                          dtype=np.float32)


def _make_sequence(root, n_frames=6, w=320, h=240):
    img_dir = os.path.join(root, "img1")
    gt_dir = os.path.join(root, "gt")
    os.makedirs(img_dir)
    os.makedirs(gt_dir)
    gt_lines = []
    bw, bh = 30, 60
    for f in range(1, n_frames + 1):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # identity 1: red, moving right;  identity 2: blue, moving left
        x1 = 20 + 10 * (f - 1)
        x2 = w - 50 - 10 * (f - 1)
        y = 80
        cv2.rectangle(img, (x1, y), (x1 + bw, y + bh), (0, 0, 255), -1)   # red (BGR)
        cv2.rectangle(img, (x2, y), (x2 + bw, y + bh), (255, 0, 0), -1)   # blue
        cv2.imwrite(os.path.join(img_dir, "%06d.jpg" % f), img)
        gt_lines.append("%d,1,%d,%d,%d,%d,1,1,1" % (f, x1, y, bw, bh))
        gt_lines.append("%d,2,%d,%d,%d,%d,1,1,1" % (f, x2, y, bw, bh))
    with open(os.path.join(gt_dir, "gt.txt"), "w") as fh:
        fh.write("\n".join(gt_lines) + "\n")
    with open(os.path.join(root, "seqinfo.ini"), "w") as fh:
        fh.write("[Sequence]\nname=SYNTH\nimDir=img1\nframeRate=30\n"
                 "seqLength=%d\nimWidth=%d\nimHeight=%d\nimExt=.jpg\n" % (n_frames, w, h))


def main():
    from detectors.gt_detector import GtDetector
    from pipeline import FrameSource, MotResultWriter, TrackingRunner

    tmp = tempfile.mkdtemp(prefix="synthmot_")
    seq_dir = os.path.join(tmp, "SYNTH")
    os.makedirs(seq_dir)
    _make_sequence(seq_dir)

    fs = FrameSource(seq_dir)
    assert len(fs) == 6 and fs.image_size == (240, 320), (len(fs), fs.image_size)

    det = GtDetector({"name": "gt", "gt_file": os.path.join(seq_dir, "gt", "gt.txt"),
                      "pedestrian_classes": [1], "person_class_id": 1}, device="cpu")
    reid = StubReID({}, device="cpu")
    cfg = {"tracker": {"n_init": 2, "max_age": 30, "max_cosine_distance": 0.2,
                       "nn_budget": 50}, "reid": {}}
    out = os.path.join(tmp, "SYNTH.txt")
    runner = TrackingRunner(det, reid, cfg, mode="gtbox")
    stats = runner.run_sequence(fs, writer=MotResultWriter(out))

    assert os.path.exists(out), "no output file"
    rows = [l for l in open(out).read().splitlines() if l.strip()]
    ids = sorted({int(l.split(",")[1]) for l in rows})
    print("frames=%d rows=%d ids=%s fps=%.1f" % (stats.n_frames, len(rows), ids, stats.fps))
    assert len(rows) >= 4, "expected several tracked rows, got %d" % len(rows)
    assert len(ids) == 2, "expected exactly 2 identities, got %s" % ids
    print("PASS test_pipeline_e2e")


if __name__ == "__main__":
    main()
