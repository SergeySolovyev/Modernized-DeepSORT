"""REID-for-tracker study: GT boxes (SORT detection disabled), vary only the REID model.

Runs each REID in gtbox mode over all sequences, computes per-video HOTA via TrackEval,
and prints a table isolating the appearance model's effect on tracking.

  python -m eval.reid_eval --reids osnet_x1_0 osnet_ain_x1_0 resnet50 timm_mobilenet mars
"""
import argparse

import numpy as np

from eval.common import EVAL_SEQUENCES, benchmark_of, tracker_name
from eval.run_tracking import run_sequence
from eval.trackeval_runner import run_trackeval_per_seq


def main():
    ap = argparse.ArgumentParser(description="REID-only HOTA (GT boxes)")
    ap.add_argument("--reids", nargs="+", required=True)
    ap.add_argument("--sequences", nargs="*", default=EVAL_SEQUENCES)
    ap.add_argument("--data-root", default="data/MOT")
    ap.add_argument("--trackeval-root", default="third_party/TrackEval")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    table = {}  # reid -> {seq -> HOTA}
    for reid in args.reids:
        for seq in args.sequences:
            try:
                run_sequence("gt", reid, seq, "gtbox", args.data_root, args.device)
            except Exception as exc:
                print("ERROR track %s/%s: %r" % (reid, seq, exc))
        per_seq = {}
        tname = tracker_name("gt", reid, "gtbox")
        for bench in sorted({benchmark_of(s) for s in args.sequences}):
            seqs = [s for s in args.sequences if benchmark_of(s) == bench]
            try:
                res = run_trackeval_per_seq(bench, tname, seqs, args.trackeval_root)
                for s, m in res.items():
                    per_seq[s] = m.get("HOTA", float("nan"))
            except Exception as exc:
                print("ERROR trackeval %s/%s: %r" % (reid, bench, exc))
        table[reid] = per_seq

    # print HOTA-by-reid table
    header = "%-16s " % "REID" + " ".join("%-14s" % s for s in args.sequences) + " %-7s" % "MEAN"
    print("\n" + header)
    for reid, per_seq in table.items():
        vals = [per_seq.get(s, float("nan")) for s in args.sequences]
        mean = float(np.nanmean(vals)) if vals else float("nan")
        print("%-16s " % reid + " ".join("%-14.2f" % v for v in vals) + " %-7.2f" % mean)


if __name__ == "__main__":
    main()
