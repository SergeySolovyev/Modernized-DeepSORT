"""Reproduce the UNMODIFIED DeepSORT baseline (mandatory Step 1).

Original pipeline: the provided MOTChallenge detections (det/det.txt) + the original
mars-small128 appearance model + the unchanged tracker. Results are written into the
TrackEval layout as the `baseline` tracker so HOTA is directly comparable to the modern
configs. Requires TensorFlow + third_party/deep_sort_data/mars-small128.pb (see data/README).

  python -m eval.run_baseline --mars third_party/deep_sort_data/mars-small128.pb
  python -m eval.trackeval_runner --benchmark MOT15 --trackers baseline
  python -m eval.trackeval_runner --benchmark MOT16 --trackers baseline

Default tracker params are the original DeepSORT settings (max_cosine_distance=0.2,
nn_budget=100, min_confidence=0.3, nms_max_overlap=1.0).
"""
import argparse
import os

import deep_sort_app
from eval import journal
from eval.common import EVAL_SEQUENCES, benchmark_of, tracker_result_path


def main():
    ap = argparse.ArgumentParser(description="Unmodified DeepSORT baseline (Step 1)")
    ap.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--mars", default="third_party/deep_sort_data/mars-small128.pb")
    ap.add_argument("--npy-dir", default="data/baseline_npy")
    ap.add_argument("--tracker-name", default="baseline")
    ap.add_argument("--min-confidence", type=float, default=0.3)
    ap.add_argument("--nms-max-overlap", type=float, default=1.0)
    ap.add_argument("--min-detection-height", type=int, default=0)
    ap.add_argument("--max-cosine-distance", type=float, default=0.2)
    ap.add_argument("--nn-budget", type=int, default=100)
    args = ap.parse_args()

    # Lazy import (pulls in TensorFlow) so the module stays importable without TF.
    from tools import generate_detections as gd

    encoder = gd.create_box_encoder(args.mars, batch_size=32)
    os.makedirs(args.npy_dir, exist_ok=True)
    # Encodes mars features for the provided detections of every sequence under data-root.
    gd.generate_detections(encoder, args.data_root, args.npy_dir)

    for seq in args.sequences:
        seq_dir = os.path.join(args.data_root, seq)
        npy = os.path.join(args.npy_dir, seq + ".npy")
        if not os.path.exists(npy):
            print("  WARNING: no detections .npy for %s (need det/det.txt)" % seq)
            continue
        out = tracker_result_path(benchmark_of(seq), args.tracker_name, seq)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        deep_sort_app.run(seq_dir, npy, out, args.min_confidence, args.nms_max_overlap,
                          args.min_detection_height, args.max_cosine_distance,
                          args.nn_budget, False)
        print("baseline: %s -> %s" % (seq, out))
        journal.append_row({"component": "baseline", "detector": "provided_det",
                            "reid": "mars", "seq": seq, "tracker": args.tracker_name})


if __name__ == "__main__":
    main()
