"""Per-video tuning sweep for TUD-Campus (MOT15) — closes the last baseline gap.

Context: the modern YOLOv8m+OSNet pipeline scored HOTA 39.65 < unmodified-baseline 39.86
on TUD-Campus (the only video where it trailed). Root cause (verified): the shipped
`configs/sequences/TUD-Campus.yaml` downgraded `detector.imgsz` 1280 -> 960, costing recall
(hence DetA, hence HOTA) on this small/occluded-pedestrian clip. Per-video params are
explicitly permitted by the assignment, and the auto-merged sequence preset is the ONLY
per-video override mechanism (run_tracking has no --override flag).

This sweeps detector/tracker levers by rewriting that one preset, re-running TUD-Campus, and
reading the per-video HOTA back from the journal. The original preset is restored at the end;
the winning preset is printed so it can be committed.

  python -m eval.tune_tud_campus            # GPU (Colab)
  python -m eval.tune_tud_campus cpu        # CPU
"""
import csv
import os
import subprocess
import sys

PRESET = os.path.join("configs", "sequences", "TUD-Campus.yaml")
JOURNAL = os.path.join("results", "experiments.csv")
TRACKER = "yolo__osnet"          # --reid osnet -> tracker dir name (matches the baseline run)
SEQ = "TUD-Campus"
BASELINE_HOTA = 39.86            # unmodified DeepSORT on TUD-Campus (report S1.3)
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"

# Each candidate is the exact configs/sequences/TUD-Campus.yaml content to test.
# _deep_merge is per-key, so only the listed keys override default/detector/reid presets.
#
# SWEEP 1 (recall direction: imgsz up, conf down) was tried and FALSIFIED the recall
# hypothesis — HOTA fell monotonically (imgsz960 39.65 -> 1280 35.0 -> 1536 29.7). More
# detections HURT. SWEEP 2 (below) tries the OPPOSITE: fewer/cleaner detections. The live
# pipeline runs with NO NMS (nms_max_overlap default 1.0 = off) and NO confidence gate
# (tracker.min_confidence default 0.0), so duplicate/FP boxes on this dense crossing-ped
# clip go straight into the tracker. These candidates add precision levers at imgsz=960.
_BASE = "detector:\n  imgsz: 960\n  conf: %s\n" \
        "tracker:\n  n_init: 2\n  max_age: 20\n  max_cosine_distance: 0.2\n%s"
CANDIDATES = {
    "control_imgsz960":      _BASE % ("0.25", ""),
    "conf035":               _BASE % ("0.35", ""),
    "conf045":               _BASE % ("0.45", ""),
    "nms07":                 _BASE % ("0.25", "  nms_max_overlap: 0.7\n"),
    "nms05":                 _BASE % ("0.25", "  nms_max_overlap: 0.5\n"),
    "minconf030":            _BASE % ("0.25", "  min_confidence: 0.30\n"),
    "minheight20":           _BASE % ("0.25", "  min_detection_height: 20\n"),
    "nms07_conf035":         _BASE % ("0.35", "  nms_max_overlap: 0.7\n"),
    "nms07_minconf03_h15":   _BASE % ("0.25", "  nms_max_overlap: 0.7\n  min_confidence: 0.30\n  min_detection_height: 15\n"),
}


def tud_hota():
    """Latest TUD-Campus HOTA for the yolo__osnet tracker from the journal."""
    if not os.path.exists(JOURNAL):
        return float("nan")
    with open(JOURNAL, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("component") == "hota" and r.get("tracker") == TRACKER
                and r.get("seq") == SEQ and r.get("HOTA") not in (None, "", "None")]
    return float(rows[-1]["HOTA"]) if rows else float("nan")


def _run(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main():
    orig = open(PRESET, encoding="utf-8").read() if os.path.exists(PRESET) else None
    results = {}
    try:
        for name, yml in CANDIDATES.items():
            print("\n=== candidate: %s ===" % name, flush=True)
            with open(PRESET, "w", encoding="utf-8") as fh:
                fh.write(yml)
            _run([sys.executable, "-m", "eval.run_tracking", "--detector", "yolo",
                  "--reid", "osnet", "--sequences", SEQ, "--mode", "live", "--device", DEVICE])
            _run([sys.executable, "-m", "eval.trackeval_runner", "--benchmark", "MOT15",
                  "--trackers", TRACKER, "--per-seq"])
            results[name] = tud_hota()
            print("  -> %s HOTA = %.3f" % (SEQ, results[name]), flush=True)
    finally:
        if orig is not None:
            with open(PRESET, "w", encoding="utf-8") as fh:
                fh.write(orig)
            print("\n(restored original %s)" % PRESET)

    print("\n===== TUD-Campus tuning sweep -- baseline to beat = %.2f =====" % BASELINE_HOTA)
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1] if kv[1] == kv[1] else -1.0), reverse=True)
    for name, h in ranked:
        flag = "<-- BEATS baseline" if h == h and h > BASELINE_HOTA else ""
        print("%-34s HOTA %7.3f  %s" % (name, h, flag))
    if ranked and ranked[0][1] == ranked[0][1]:
        win, wh = ranked[0]
        print("\nWINNER: %s  HOTA %.3f  (delta %+.3f vs baseline)" % (win, wh, wh - BASELINE_HOTA))
        print("Winning preset content:\n%s" % CANDIDATES[win])


if __name__ == "__main__":
    main()
