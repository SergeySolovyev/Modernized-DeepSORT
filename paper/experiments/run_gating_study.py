"""Gating study for the paper "When More Detections Hurt".

For every evaluation video, sweep the detector confidence threshold (the recall/precision lever)
and run a gating-restoration ablation (restore the original DeepSORT min_confidence + NMS gate that
the modern live pipeline drops). On two dense low-resolution clips also sweep detector input size
(an orthogonal recall lever). For each configuration: write a per-sequence preset, run the live
tracker (YOLOv8m + OSNet), score HOTA with TrackEval, and append one row to a CSV.

Resilient: each result is written immediately and configurations already in the CSV are skipped, so
the run can be interrupted (Colab disconnect) and resumed by re-running. Original presets are
restored at the end.

  python paper/experiments/run_gating_study.py            # cuda
  python paper/experiments/run_gating_study.py cpu
"""
import csv
import os
import subprocess
import sys

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
OUT = os.path.join("paper", "experiments", "gating_study.csv")
JOURNAL = os.path.join("results", "experiments.csv")
PRESET_DIR = os.path.join("configs", "sequences")
TRACKER = "yolo__osnet"

SEQ_BENCH = [("TUD-Campus", "MOT15"), ("TUD-Stadtmitte", "MOT15"), ("KITTI-17", "MOT15"),
             ("PETS09-S2L1", "MOT15"), ("MOT16-09", "MOT16"), ("MOT16-11", "MOT16")]
DENSE = {"TUD-Campus", "KITTI-17"}            # low detector-precision clips
FIELDS = ["seq", "benchmark", "label", "conf", "imgsz", "nms_max_overlap", "min_confidence", "HOTA"]


def preset(conf, imgsz=1280, nms=1.0, minconf=0.0):
    return ("detector:\n  imgsz: %d\n  conf: %s\n"
            "tracker:\n  n_init: 2\n  max_age: 20\n  max_cosine_distance: 0.2\n"
            "  nms_max_overlap: %s\n  min_confidence: %s\n" % (imgsz, conf, nms, minconf))


def build_grid():
    grid = []  # (seq, bench, label, conf, imgsz, nms, minconf, yaml)
    for seq, bench in SEQ_BENCH:
        for c in (0.15, 0.25, 0.35, 0.45):     # confidence sweep (recall<->precision)
            grid.append((seq, bench, "conf%03d" % int(round(c * 100)), c, 1280, 1.0, 0.0, preset(c)))
        # gating-restoration ablation: original DeepSORT gate (min_confidence 0.3 + NMS 0.7)
        grid.append((seq, bench, "restored_gate", 0.25, 1280, 0.7, 0.3, preset(0.25, nms=0.7, minconf=0.3)))
        if seq in DENSE:                        # orthogonal recall lever: input size
            for s in (960, 1536):
                grid.append((seq, bench, "imgsz%d" % s, 0.25, s, 1.0, 0.0, preset(0.25, imgsz=s)))
    return grid


def done_set():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="", encoding="utf-8") as fh:
        return {(r["seq"], r["label"]) for r in csv.DictReader(fh)}


def append_row(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def latest_hota(seq):
    if not os.path.exists(JOURNAL):
        return float("nan")
    with open(JOURNAL, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("component") == "hota" and r.get("tracker") == TRACKER
                and r.get("seq") == seq and r.get("HOTA") not in (None, "", "None")]
    return float(rows[-1]["HOTA"]) if rows else float("nan")


def run(cmd):
    print("  $ python -m " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m"] + cmd, check=False)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    grid = build_grid()
    done = done_set()
    backup = {}
    for seq, _ in SEQ_BENCH:
        p = os.path.join(PRESET_DIR, "%s.yaml" % seq)
        backup[seq] = open(p, encoding="utf-8").read() if os.path.exists(p) else None
    try:
        for i, (seq, bench, label, conf, imgsz, nms, minconf, yml) in enumerate(grid):
            if (seq, label) in done:
                print("[%d/%d] skip %s/%s (already done)" % (i + 1, len(grid), seq, label))
                continue
            print("\n[%d/%d] %s / %s  (conf=%s imgsz=%d nms=%s min_conf=%s)"
                  % (i + 1, len(grid), seq, label, conf, imgsz, nms, minconf), flush=True)
            with open(os.path.join(PRESET_DIR, "%s.yaml" % seq), "w", encoding="utf-8") as fh:
                fh.write(yml)
            run(["eval.run_tracking", "--detector", "yolo", "--reid", "osnet",
                 "--sequences", seq, "--mode", "live", "--device", DEVICE])
            run(["eval.trackeval_runner", "--benchmark", bench, "--trackers", TRACKER, "--per-seq"])
            h = latest_hota(seq)
            append_row({"seq": seq, "benchmark": bench, "label": label, "conf": conf,
                        "imgsz": imgsz, "nms_max_overlap": nms, "min_confidence": minconf, "HOTA": h})
            print("  -> HOTA = %.3f" % h, flush=True)
    finally:
        for seq, _ in SEQ_BENCH:
            p = os.path.join(PRESET_DIR, "%s.yaml" % seq)
            if backup[seq] is not None:
                open(p, "w", encoding="utf-8").write(backup[seq])
            elif os.path.exists(p):
                os.remove(p)
        print("\n(restored original sequence presets)")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
