"""Run TrackEval to compute HOTA/MOTA/IDF1 for MOT-format tracker results.

Wraps TrackEval's scripts/run_mot_challenge.py (code at cfg.paths.trackeval_root)
over the data layout produced by data/download_mot.py and eval/run_tracking.py.

  python -m eval.trackeval_runner --benchmark MOT15 --trackers yolo__osnet_x1_0
"""
import argparse
import os
import subprocess
import sys

from eval.common import SPLIT, TE_GT, TE_TRACKERS, benchmark_split, seqmap_path


def _parse_summary(path):
    """Parse a TrackEval pedestrian_summary.txt (2 lines: names, then values)."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if len(lines) < 2:
        return {}
    keys = lines[0].split()
    vals = lines[1].split()
    out = {}
    for k, v in zip(keys, vals):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def run_trackeval(benchmark, trackers, trackeval_root="third_party/TrackEval",
                  metrics=("HOTA", "CLEAR", "Identity"), seqmap=None, do_preproc=None):
    """Run TrackEval; return {tracker: {HOTA, MOTA, IDF1, DetA, AssA, ...}}."""
    script = os.path.join(trackeval_root, "scripts", "run_mot_challenge.py")
    if not os.path.exists(script):
        raise FileNotFoundError("TrackEval not found at %s (clone JonathonLuiten/TrackEval)" % script)
    if do_preproc is None:
        do_preproc = benchmark != "MOT15"   # MOT15 gt lacks preproc info
    if seqmap is None:
        seqmap = seqmap_path(benchmark)

    cmd = [sys.executable, script,
           "--BENCHMARK", benchmark, "--SPLIT_TO_EVAL", SPLIT,
           "--GT_FOLDER", TE_GT, "--TRACKERS_FOLDER", TE_TRACKERS,
           "--TRACKERS_TO_EVAL", *trackers,
           "--METRICS", *metrics,
           "--USE_PARALLEL", "False", "--DO_PREPROC", str(bool(do_preproc)),
           "--SEQMAP_FILE", seqmap, "--PRINT_RESULTS", "True"]
    print("[trackeval]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results = {}
    for tracker in trackers:
        summ = os.path.join(TE_TRACKERS, benchmark_split(benchmark), tracker,
                            "pedestrian_summary.txt")
        results[tracker] = _parse_summary(summ)
    return results


def run_trackeval_per_seq(benchmark, tracker, sequences, trackeval_root="third_party/TrackEval"):
    """Per-video HOTA by running TrackEval with a temp 1-sequence seqmap each time.

    Returns {seq: {HOTA, MOTA, IDF1, ...}}. Reliable per-video numbers for the report.
    """
    out = {}
    seqmap_dir = os.path.join(TE_GT, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    for seq in sequences:
        tmp = os.path.join(seqmap_dir, "_tmp_%s.txt" % seq)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("name\n%s\n" % seq)
        try:
            res = run_trackeval(benchmark, [tracker], trackeval_root, seqmap=tmp)
            out[seq] = res.get(tracker, {})
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute HOTA via TrackEval")
    ap.add_argument("--benchmark", required=True, choices=["MOT15", "MOT16"])
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--trackeval-root", default="third_party/TrackEval")
    args = ap.parse_args()
    res = run_trackeval(args.benchmark, args.trackers, args.trackeval_root)
    for tracker, m in res.items():
        print("%-28s HOTA=%.2f MOTA=%.2f IDF1=%.2f DetA=%.2f AssA=%.2f"
              % (tracker, m.get("HOTA", float("nan")), m.get("MOTA", float("nan")),
                 m.get("IDF1", float("nan")), m.get("DetA", float("nan")),
                 m.get("AssA", float("nan"))))


if __name__ == "__main__":
    main()
