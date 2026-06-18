"""Run TrackEval to compute HOTA/MOTA/IDF1 for MOT-format tracker results.

Wraps TrackEval's scripts/run_mot_challenge.py (code at cfg.paths.trackeval_root)
over the data layout produced by data/download_mot.py and eval/run_tracking.py.

  python -m eval.trackeval_runner --benchmark MOT15 --trackers yolo__osnet_x1_0
"""
import argparse
import glob
import os
import re
import subprocess
import sys

from eval import journal
from eval.common import SPLIT, TE_GT, TE_TRACKERS, benchmark_split, seqmap_path

_METRIC_KEYS = ("HOTA", "MOTA", "IDF1", "DetA", "AssA")


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


def _patch_trackeval_numpy(root):
    """TrackEval uses np.float/np.int/np.bool, removed in numpy 2.x. Rewrite in place
    so TrackEval runs on modern numpy (idempotent; word-boundaried to spare float64 etc.)."""
    for path in glob.glob(os.path.join(root, "trackeval", "**", "*.py"), recursive=True):
        try:
            with open(path, encoding="utf-8") as fh:
                src = fh.read()
            new = re.sub(r"np\.float\b", "float", src)
            new = re.sub(r"np\.int\b", "int", new)
            new = re.sub(r"np\.bool\b", "bool", new)
            if new != src:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(new)
        except Exception:
            pass


def run_trackeval(benchmark, trackers, trackeval_root="third_party/TrackEval",
                  metrics=("HOTA", "CLEAR", "Identity"), do_preproc=None):
    """Run TrackEval; return {tracker: {HOTA, MOTA, IDF1, DetA, AssA, ...}}.

    Uses TrackEval's DEFAULT seqmap at GT_FOLDER/seqmaps/<BENCH>-train.txt (written by
    data/download_mot.py). We deliberately do NOT pass --SEQMAP_FILE: TrackEval registers
    that arg with nargs='+', so an explicit value arrives as a list and crashes seqmap
    handling (os.path.isfile(list) -> TypeError).
    """
    script = os.path.join(trackeval_root, "scripts", "run_mot_challenge.py")
    if not os.path.exists(script):
        raise FileNotFoundError("TrackEval not found at %s (clone JonathonLuiten/TrackEval)" % script)
    _patch_trackeval_numpy(trackeval_root)   # numpy 2.x compatibility
    if do_preproc is None:
        do_preproc = benchmark != "MOT15"   # MOT15 gt lacks preproc info

    cmd = [sys.executable, script,
           "--BENCHMARK", benchmark, "--SPLIT_TO_EVAL", SPLIT,
           "--GT_FOLDER", TE_GT, "--TRACKERS_FOLDER", TE_TRACKERS,
           "--TRACKERS_TO_EVAL", *trackers,
           "--METRICS", *metrics,
           "--USE_PARALLEL", "False", "--DO_PREPROC", str(bool(do_preproc)),
           "--PRINT_RESULTS", "True"]
    print("[trackeval]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    results = {}
    for tracker in trackers:
        summ = os.path.join(TE_TRACKERS, benchmark_split(benchmark), tracker,
                            "pedestrian_summary.txt")
        results[tracker] = _parse_summary(summ)
    return results


def run_trackeval_per_seq(benchmark, tracker, sequences, trackeval_root="third_party/TrackEval"):
    """Per-video HOTA: temporarily restrict the DEFAULT seqmap to one sequence per run.

    TrackEval auto-reads GT_FOLDER/seqmaps/<BENCH>-train.txt; since --SEQMAP_FILE can't be
    passed safely, we overwrite that file with a single sequence (backing up + restoring the
    original). Returns {seq: {HOTA, MOTA, IDF1, ...}} — reliable per-video numbers.
    """
    out = {}
    seqmap = seqmap_path(benchmark)
    os.makedirs(os.path.dirname(seqmap), exist_ok=True)
    backup = None
    if os.path.exists(seqmap):
        with open(seqmap, encoding="utf-8") as fh:
            backup = fh.read()
    try:
        for seq in sequences:
            with open(seqmap, "w", encoding="utf-8") as fh:
                fh.write("name\n%s\n" % seq)
            res = run_trackeval(benchmark, [tracker], trackeval_root)
            m = res.get(tracker, {})
            out[seq] = m
            journal.append_row({"component": "hota", "benchmark": benchmark,
                                "tracker": tracker, "seq": seq,
                                **{k: m.get(k) for k in _METRIC_KEYS}})
    finally:
        if backup is not None:
            with open(seqmap, "w", encoding="utf-8") as fh:
                fh.write(backup)
    return out


def main():
    ap = argparse.ArgumentParser(description="Compute HOTA via TrackEval")
    ap.add_argument("--benchmark", required=True, choices=["MOT15", "MOT16"])
    ap.add_argument("--trackers", nargs="+", required=True)
    ap.add_argument("--trackeval-root", default="third_party/TrackEval")
    ap.add_argument("--per-seq", action="store_true",
                    help="evaluate per-sequence and journal HOTA (feeds eval.summarize's report table)")
    args = ap.parse_args()

    if args.per_seq:
        from eval.common import SEQ_BENCHMARK
        seqs = [s for s in SEQ_BENCHMARK if SEQ_BENCHMARK[s] == args.benchmark]
        for tracker in args.trackers:
            try:
                res = run_trackeval_per_seq(args.benchmark, tracker, seqs, args.trackeval_root)
                for seq, m in res.items():
                    print("%-28s %-16s HOTA=%.2f" % (tracker, seq, m.get("HOTA", float("nan"))))
            except Exception as exc:
                print("ERROR %s: %r" % (tracker, exc))
        return

    res = run_trackeval(args.benchmark, args.trackers, args.trackeval_root)
    for tracker, m in res.items():
        print("%-28s HOTA=%.2f MOTA=%.2f IDF1=%.2f DetA=%.2f AssA=%.2f"
              % (tracker, m.get("HOTA", float("nan")), m.get("MOTA", float("nan")),
                 m.get("IDF1", float("nan")), m.get("DetA", float("nan")),
                 m.get("AssA", float("nan"))))


if __name__ == "__main__":
    main()
