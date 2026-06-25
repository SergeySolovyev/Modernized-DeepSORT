# Results & Reproducibility

Modernized DeepSORT - person tracking with pluggable detectors and REID.
This note records the verified results on Colab Pro and the exact, copy-pasteable
commands to reproduce them. All HOTA numbers use the TrackEval protocol over the six
evaluation videos: TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 (MOT15) and
MOT16-09, MOT16-11 (MOT16).

## 1. Main result (verified, live run)

Mean HOTA over the six videos (per-sequence rows in `results/experiments.csv`):

| System | mean HOTA | Delta | per-video wins | FPS | real-time |
|--------|-----------|-------|----------------|-----|-----------|
| Unmodified DeepSORT baseline (provided dets + mars-small128) | 40.17 | - | - | - | - |
| Modern: YOLOv8m + OSNet | 52.54 | +12.37 | 6 / 6 | 9.87 | yes |
| Modern: YOLOv8m + timm (no-boxmot fallback) | 47.70 | +7.53 | 5 / 6 | - | - |

YOLOv8m + OSNet exceeds the unmodified baseline on every one of the six videos
(smallest margin +5.34 on KITTI-17) while running at 9.87 FPS on a Colab T4, meeting
the >= 5 FPS real-time requirement. The gain is driven by detection quality and a
stronger appearance metric.

Per-video HOTA (baseline -> YOLOv8m+OSNet):
TUD-Campus 39.86 -> 46.98 ; TUD-Stadtmitte 36.75 -> 59.62 ; KITTI-17 43.41 -> 48.75 ;
PETS09-S2L1 44.84 -> 60.28 ; MOT16-09 36.24 -> 48.57 ; MOT16-11 39.95 -> 51.07.

REID-only HOTA on GT boxes (appearance metric isolated, SORT detection disabled):
OSNet 89.93, timm 89.22, osnet_ain 87.33, osnet_fast 86.13.

Detector P/R/F1 vs GT (IoU>=0.5, mean over six videos): YOLOv8m 0.721 / 0.828 / 0.765 ;
YOLOv8m-seg 0.727 / 0.827 / 0.768 (detector + segmentation refreshed 2026-06-25, current YOLOv8m weights).

## 2. Reproduce end-to-end on Colab Pro (copy-paste)

Runtime: GPU (T4 or better). The notebook `notebooks/DeepSORT_Modern.ipynb` runs the
whole study top to bottom (Runtime -> Run all). The explicit commands are:

```bash
# --- Clone + enter ---
git clone https://github.com/SergeySolovyev/Modernized-DeepSORT.git
cd Modernized-DeepSORT && git pull -q

# --- Install. Core MUST succeed; OSNet (boxmot) and mmdet are guarded so a ---
# --- runtime build failure cannot break the core pipeline. ---
pip -q install -r requirements-modern.txt
pip -q install boxmot || echo 'boxmot (OSNet) failed -> timm REID fallback'
pip -q install -U openmim && mim install mmengine 'mmcv>=2.0' mmdet || echo 'mmdet skipped'
git clone -q https://github.com/JonathonLuiten/TrackEval third_party/TrackEval || true
pip -q install -e third_party/TrackEval

# --- Data ---
python -m data.download_mot

# --- Baseline: unmodified DeepSORT (provided dets + mars-small128, auto-downloaded) ---
python -m eval.run_baseline --mars third_party/deep_sort_data/mars-small128.pb
python -m eval.trackeval_runner --benchmark MOT15 --trackers baseline --per-seq
python -m eval.trackeval_runner --benchmark MOT16 --trackers baseline --per-seq

# --- Modern best combo: YOLOv8 + OSNet -> live tracking -> HOTA ---
python -m eval.run_tracking --detector yolo --reid osnet --device cuda
python -m eval.trackeval_runner --benchmark MOT15 --trackers yolo__osnet --per-seq
python -m eval.trackeval_runner --benchmark MOT16 --trackers yolo__osnet --per-seq

# --- Real-time FPS check (>= 5 FPS) ---
python -m eval.fps_bench --detector yolo --reid osnet --sequence MOT16-09 --device cuda

# --- Per-video HOTA table (tracker x video + Mean + Delta vs baseline) ---
python -m eval.summarize

# --- Detector + segmentation studies, and the standalone body-REID identity system ---
python -m eval.fill_report_numbers          # detector P/R/F1 + yolo_seg segmentation HOTA
python -m eval.finalize_study               # standalone body-REID clustering + parameter sweep
```

timm is the no-boxmot fallback: pass `--reid timm` anywhere `--reid osnet` appears.

## 3. Known-good environment notes

Runtime realities discovered by running on Colab Pro; the code handles each automatically.

- OSNet now builds. boxmot (BoxMOT v19.0.0) installs on the current Colab runtime, so
  OSNet is the headline REID; timm remains the guarded fallback if boxmot ever fails.
- MOT data mirror. `motchallenge.net` is unreachable from Colab; `data/mot.py` defaults to
  the PaddleDetection / Baidu mirror, preserving the standard MOTChallenge directory layout.
- mars-small128.pb auto-download. `eval/run_baseline.py` fetches the baseline appearance model.
- TrackEval numpy-2.x patch. `eval/trackeval_runner.py` auto-patches `np.float` / `np.int` /
  `np.bool` (removed in numpy 2.x) before running.
- TrackEval empty-tracker guard. Only pass trackers (`--trackers`) that produced result files.

## 4. Committed artifacts

The detector and segmentation studies were refreshed on the 2026-06-25 Colab run; the standalone
body-REID clustering reproduced the committed numbers exactly (embedding FMI 0.630, assignment FMI
0.242, 21,105 crops / 140 identities). The following are versioned in the repository for traceability:

- `results/experiments.csv` - the experiment journal (detector P/R/F1, per-video HOTA for
  baseline / yolo__osnet / yolo__timm / GT-box REID studies, FPS).
- `report/results_table.md` - the per-video HOTA table (tracker x video + Mean + Delta).
- `report/report.pdf` - the full report (built from `report/latex/main.tex`).
- `overlays/baseline_MOT16-09.mp4`, `overlays/best_MOT16-09.mp4` - qualitative overlays.
