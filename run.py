"""Single CLI entrypoint for the modernized DeepSORT.

Examples
--------
    # live tracking, write MOT results
    python run.py --detector yolo --reid osnet_x1_0 --sequence MOT16-09

    # REID-only mode (GT boxes, SORT detection disabled) to isolate appearance
    python run.py --detector gt --reid osnet_x1_0 --sequence TUD-Campus --mode gtbox

    # per-run overrides
    python run.py --detector yolo --reid resnet50 --sequence KITTI-17 \
        --override tracker.max_cosine_distance=0.3 detector.imgsz=960
"""
import argparse
import json
import os

from config import build_config
from detectors.registry import build_detector
from reid.registry import build_reid
from pipeline import FrameSource, MotResultWriter, TrackingRunner


def resolve_sequence_dir(cfg, args):
    if args.sequence_dir:
        return args.sequence_dir
    if not args.sequence:
        raise ValueError("Provide --sequence <name> or --sequence-dir <path>")
    return os.path.join(cfg["paths"]["data_root"], args.sequence)


def parse_args():
    ap = argparse.ArgumentParser(description="Modernized DeepSORT runner")
    ap.add_argument("--detector", default="yolo", help="detector preset (configs/detectors/*.yaml)")
    ap.add_argument("--reid", default="osnet_x1_0", help="reid preset (configs/reid/*.yaml)")
    ap.add_argument("--sequence", default=None, help="MOT sequence name (resolved under data_root)")
    ap.add_argument("--sequence-dir", default=None, help="explicit path to a MOT sequence directory")
    ap.add_argument("--mode", default="live", choices=["live", "gtbox"])
    ap.add_argument("--output", default=None, help="MOT results output file")
    ap.add_argument("--device", default=None, help="cuda | cpu (defaults to config)")
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value config overrides")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = build_config(detector=args.detector, reid=args.reid,
                       sequence=args.sequence, overrides=args.override)
    device = args.device or cfg.get("device", "cuda")

    seq_dir = resolve_sequence_dir(cfg, args)
    frame_source = FrameSource(seq_dir)

    # In gtbox mode, point GtDetector at this sequence's gt.txt unless overridden.
    if cfg["detector"].get("name") == "gt" and not cfg["detector"].get("gt_file"):
        cfg["detector"]["gt_file"] = os.path.join(seq_dir, "gt", "gt.txt")

    detector = build_detector(cfg, device)
    reid = build_reid(cfg, device)

    if args.output:
        out = args.output
    else:
        tag = "%s__%s" % (args.detector, args.reid)
        out = os.path.join(cfg["paths"]["results_root"], tag, frame_source.name + ".txt")

    writer = MotResultWriter(out)
    runner = TrackingRunner(detector, reid, cfg, mode=args.mode)
    stats = runner.run_sequence(frame_source, writer=writer)

    print(json.dumps(stats.to_dict(), indent=2))
    print("results ->", out)


if __name__ == "__main__":
    main()
