"""Fill the two remaining (pending) report tables: detector P/R/F1 (S2.2) and segmentation (S5).

Runs det_eval for each detector and yolo_seg live tracking, scores HOTA, then prints clean mean
summaries to paste into report.md. Detectors whose backend isn't installed on the runtime are
reported as NOT AVAILABLE (honest negative result) rather than crashing the run.

  python -m eval.fill_report_numbers            # cuda
  python -m eval.fill_report_numbers cpu
"""
import csv
import os
import statistics
import subprocess
import sys

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
JOURNAL = os.path.join("results", "experiments.csv")
DETECTORS = ["yolo", "yolo_seg", "nanodet", "mmdet"]


def _run(cmd):
    print("  $ python -m " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m"] + cmd, check=False)


def det_mean(det):
    path = os.path.join("results", "det_eval_%s.csv" % det)
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return None
    return (statistics.mean(float(r["precision"]) for r in rows),
            statistics.mean(float(r["recall"]) for r in rows),
            statistics.mean(float(r["f1"]) for r in rows),
            len(rows))


def _journal(**filt):
    if not os.path.exists(JOURNAL):
        return []
    with open(JOURNAL, newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if all(r.get(k) == v for k, v in filt.items())]


def hota_mean(tracker):
    by_seq = {}
    for r in _journal(component="hota", tracker=tracker):
        if r.get("HOTA") not in (None, "", "None"):
            by_seq[r["seq"]] = float(r["HOTA"])          # last value per seq wins
    if not by_seq:
        return float("nan"), {}
    return sum(by_seq.values()) / len(by_seq), by_seq


def fps_mean(detector):
    by_seq = {}
    for r in _journal(component="track", detector=detector):
        if r.get("fps") not in (None, "", "None"):
            by_seq[r.get("sequence", "?")] = float(r["fps"])
    return sum(by_seq.values()) / len(by_seq) if by_seq else float("nan")


def main():
    print("\n########## DETECTOR P/R/F1 vs GT @ IoU>=0.5 ##########")
    for det in DETECTORS:
        print("\n--- det_eval %s ---" % det)
        _run(["eval.det_eval", "--detector", det, "--device", DEVICE])

    print("\n########## SEGMENTATION live tracking (yolo_seg + osnet) ##########")
    _run(["eval.run_tracking", "--detector", "yolo_seg", "--reid", "osnet", "--device", DEVICE])
    _run(["eval.trackeval_runner", "--benchmark", "MOT15", "--trackers", "yolo_seg__osnet", "--per-seq"])
    _run(["eval.trackeval_runner", "--benchmark", "MOT16", "--trackers", "yolo_seg__osnet", "--per-seq"])

    print("\n\n================= SUMMARY (paste into report) =================")
    print("\n[S2.2 Detector P/R/F1 -- mean over sequences]")
    for det in DETECTORS:
        m = det_mean(det)
        if m:
            print("  %-9s P=%.3f  R=%.3f  F1=%.3f   (%d seqs)" % (det, m[0], m[1], m[2], m[3]))
        else:
            print("  %-9s NOT AVAILABLE (backend not installed on this runtime)" % det)

    print("\n[S5 Segmentation: yolo box vs yolo_seg]")
    yh, _ = hota_mean("yolo__osnet")
    sh, sh_by = hota_mean("yolo_seg__osnet")
    yf, sf = det_mean("yolo"), det_mean("yolo_seg")
    print("  yolo (box):  meanF1=%s  meanHOTA=%.3f  meanFPS=%.2f" % (
        ("%.3f" % yf[2]) if yf else "n/a", yh, fps_mean("yolo")))
    print("  yolo_seg:    meanF1=%s  meanHOTA=%.3f  meanFPS=%.2f" % (
        ("%.3f" % sf[2]) if sf else "n/a", sh, fps_mean("yolo_seg")))
    print("  yolo_seg per-seq HOTA:", {k: round(v, 2) for k, v in sh_by.items()})


if __name__ == "__main__":
    main()
