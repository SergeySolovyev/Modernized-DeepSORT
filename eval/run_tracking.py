"""Run the tracker over sequences and write MOT results into the TrackEval layout.

  # live tracking with a modern detector + REID
  python -m eval.run_tracking --detector yolo --reid osnet_x1_0

  # REID-only mode: GT boxes, SORT detection disabled (isolate appearance)
  python -m eval.run_tracking --detector gt --reid osnet_x1_0 --mode gtbox
"""
import argparse
import os

from config import build_config
from detectors.registry import build_detector
from eval import journal
from eval.common import EVAL_SEQUENCES, benchmark_of, tracker_name, tracker_result_path
from pipeline import FrameSource, MotResultWriter, TrackingRunner
from reid.registry import build_reid


def run_sequence(detector_name, reid_name, seq, mode, data_root, device, body_reid=None):
    seq_dir = os.path.join(data_root, seq)
    cfg = build_config(detector=detector_name, reid=reid_name, sequence=seq)

    det_label = "gt" if mode == "gtbox" else detector_name
    if mode == "gtbox":
        cfg["detector"] = {"name": "gt", "gt_file": os.path.join(seq_dir, "gt", "gt.txt"),
                           "pedestrian_classes": [1], "person_class_id": 1}
    elif cfg["detector"].get("name") == "gt" and not cfg["detector"].get("gt_file"):
        cfg["detector"]["gt_file"] = os.path.join(seq_dir, "gt", "gt.txt")

    fs = FrameSource(seq_dir)
    detector = build_detector(cfg, device)
    reid = build_reid(cfg, device)

    bench = benchmark_of(seq)
    tname = tracker_name(det_label, reid_name, mode)
    out = tracker_result_path(bench, tname, seq)

    runner = TrackingRunner(detector, reid, cfg, mode=mode, body_reid=body_reid)
    stats = runner.run_sequence(fs, writer=MotResultWriter(out))
    journal.append_row({"component": "track", "detector": det_label, "reid": reid_name,
                        "mode": mode, **stats.to_dict()})
    print("%-16s -> %s  (%.1f FPS)" % (seq, out, stats.fps))
    return out, tname, bench, stats


def main():
    ap = argparse.ArgumentParser(description="Run tracker -> MOT results (TrackEval layout)")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--reid", default="osnet_x1_0")
    ap.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    ap.add_argument("--mode", default="live", choices=["live", "gtbox"])
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for seq in args.sequences:
        try:
            run_sequence(args.detector, args.reid, seq, args.mode, args.data_root, args.device)
        except Exception as exc:
            print("ERROR %s: %r" % (seq, exc))


if __name__ == "__main__":
    main()
