"""FPS benchmark for a (detector, reid) combo — proves the >=5 FPS real-time bar.

Times detector / REID / tracker stages separately (after warmup) over one sequence.

  python -m eval.fps_bench --detector yolo --reid osnet_x1_0 --sequence MOT16-09
"""
import argparse
import os

from config import build_config
from detectors.registry import build_detector
from eval import journal
from pipeline import FrameSource, MotResultWriter, TrackingRunner
from reid.registry import build_reid


def main():
    ap = argparse.ArgumentParser(description="FPS benchmark")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--reid", default="osnet_x1_0")
    ap.add_argument("--sequence", default="MOT16-09")
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seq_dir = os.path.join(args.data_root, args.sequence)
    cfg = build_config(detector=args.detector, reid=args.reid, sequence=args.sequence)
    if cfg["detector"].get("name") == "gt" and not cfg["detector"].get("gt_file"):
        cfg["detector"]["gt_file"] = os.path.join(seq_dir, "gt", "gt.txt")

    fs = FrameSource(seq_dir)
    detector = build_detector(cfg, args.device)
    reid = build_reid(cfg, args.device)
    detector.warmup(fs.image_size)
    reid.warmup()

    runner = TrackingRunner(detector, reid, cfg, mode="live")
    stats = runner.run_sequence(fs, writer=MotResultWriter(os.devnull))
    d = stats.to_dict()

    print("detector=%s reid=%s seq=%s" % (args.detector, args.reid, args.sequence))
    print("  overall %.2f FPS | det %.2f | reid %.2f | track %.2f"
          % (d["fps"], d["det_fps"], d["reid_fps"], d["track_fps"]))
    print("  per-frame ms: det %.1f | reid %.1f | track %.1f"
          % (d["det_ms"], d["reid_ms"], d["track_ms"]))
    print("  REAL-TIME (>=5 FPS): %s" % ("YES" if d["fps"] >= 5 else "NO"))
    journal.append_row({"component": "fps", "detector": args.detector, "reid": args.reid,
                        "realtime_5fps": d["fps"] >= 5, **d})


if __name__ == "__main__":
    main()
