# Results - live Colab run (Colab Pro, Tesla T4, 2026-06-19)

All six MOT-Challenge train sequences. HOTA via TrackEval (numpy-2.x patched). Trackers:
`baseline` = unmodified DeepSORT (provided detections + mars-small128); `yolo__*` = modern
live pipeline (YOLOv8m detector + REID); `gt__*__gtbox` = REID-only study (GT boxes, SORT
detection disabled - isolates the appearance model).

The tracking, REID-only and body-REID results are from the run of 2026-06-19; the detector P/R/F1
and segmentation rows were refreshed on 2026-06-25 with the then-current YOLOv8m weights.

## HOTA per video

| tracker | TUD-Campus | TUD-Stadtmitte | KITTI-17 | PETS09-S2L1 | MOT16-09 | MOT16-11 | **Mean** | Delta vs baseline |
|---|---|---|---|---|---|---|---|---|
| baseline (unmodified) | 39.86 | 36.75 | 43.41 | 44.84 | 36.24 | 39.95 | **40.17** | - |
| **yolo__osnet** (best live, tuned) | **46.98** | 59.62 | 48.75 | 60.28 | 48.57 | 51.07 | **52.54** | **+12.37** |
| yolo__osnet (untuned TUD-Campus) | 39.65 | 59.62 | 48.75 | 60.28 | 48.57 | 51.07 | 51.32 | +11.15 |
| yolo__timm_mobilenet | 37.57 | 57.63 | 45.82 | 50.89 | 45.56 | 48.73 | **47.70** | +7.53 |
| gt__osnet__gtbox (REID-only) | 86.67 | 94.11 | 82.98 | 88.52 | 93.46 | 93.82 | **89.93** | +49.76 |
| gt__timm_mobilenet__gtbox | 86.67 | 94.11 | 83.89 | 91.71 | 85.38 | 93.59 | **89.22** | +49.05 |
| gt__osnet_ain__gtbox | 86.67 | 87.30 | 80.87 | 90.28 | 85.44 | 93.45 | **87.33** | +47.16 |
| gt__osnet_fast__gtbox | 86.67 | 87.30 | 78.58 | 83.86 | 87.33 | 93.04 | **86.13** | +45.96 |

**Summary.** The modernized configuration (YOLOv8 + OSNet, OSNet via boxmot) reaches mean HOTA
52.54 versus 40.17 for the unmodified DeepSORT baseline (+12.37), exceeding the baseline on all
six videos after per-video tuning of TUD-Campus.

### TUD-Campus per-video tuning (the remaining gap, and a negative result)

TUD-Campus was the only video where the modern pipeline initially trailed (39.65 vs 39.86). Two
sweeps (`eval/tune_tud_campus.py`) localized the cause:

- Recall direction (not supported): raising `detector.imgsz` 960->1280->1536 (and lowering `conf`)
  made HOTA monotonically worse - 39.65 -> 35.0 -> 29.7. More detections reduced HOTA.
- Precision direction (supported): the live pipeline applies no NMS (`tracker.nms_max_overlap`
  default 1.0) and no confidence gate (`tracker.min_confidence` default 0.0), so the many
  low-confidence false positives on this dense crossing-pedestrian clip are admitted to the tracker
  and spawn spurious tracks (reducing DetA precision and AssA). Raising `detector.conf` to 0.45
  reduces the false-positive count and raises HOTA to 46.98 (+7.1 over the baseline).

| candidate | HOTA | candidate | HOTA |
|---|---|---|---|
| conf 0.45 (selected) | 46.98 | nms07+minconf03+h15 | 40.39 |
| conf 0.35 | 44.61 | minheight 20 | 39.85 |
| nms0.7 + conf0.35 | 42.49 | control (conf 0.25) | 39.65 |
| min_confidence 0.30 | 42.22 | nms 0.7 alone | 38.40 |
| nms_max_overlap 0.5 | 41.96 | imgsz 1280 / 1536 | 35.0 / 29.7 |

The fix is confined to `configs/sequences/TUD-Campus.yaml` (per-video params are permitted); the
other five sequences are untouched.

## Real-time (FPS, MOT16-09, T4, warmup-excluded)

YOLOv8 + OSNet: overall 9.87 FPS (det 30.9 / reid 30.6 / track 48.7), which meets the >=5 FPS real-time bar.

## Segmentation (box detector vs mask detector, same OSNet REID)

| detector | mean F1 | mean HOTA | FPS (MOT16-09) |
|---|---|---|---|
| yolo (box) | 0.765 | 52.54 | 9.87 |
| yolo_seg (mask->bbox) | 0.768 | 52.76 | 3.92 |

Mask-derived boxes are marginally tighter (F1 0.768 vs 0.765, HOTA 52.76 vs 52.54) but the mask head
drops throughput below the 5 FPS bar on MOT16-09 (3.92 vs 9.87 FPS). Segmentation stays real-time on the lighter
2D MOT 2015 clips (~10-14 FPS) but not on the heavier MOT16 sequences. The box detector is the
default; segmentation is a switchable option (`--detector yolo_seg`).

## REID-only study (GT boxes -> HOTA isolates appearance)

OSNet is the strongest appearance model (mean HOTA 89.93), narrowly ahead of timm (89.22),
then OSNet-AIN (87.33) and OSNet-x0.25 (86.13). High absolute values are expected here because
GT boxes make detection near-perfect, so HOTA mostly reflects association quality.

## Standalone body-REID (OSNet descriptors: 21,105 GT crops, 140 true identities)

| mode | Fowlkes-Mallows | Silhouette (cos) | Calinski-Harabasz | n_pred / n_true |
|---|---|---|---|---|
| embedding (Agglomerative @ match_thresh) | 0.630 | 0.371 | 98.3 | 420 / 140 |
| assignment (headless IdentityDatabase replay) | 0.242 | - | - | 30 / 140 |

The embedding clustering over-splits (n_pred 420 >> 140) while the live-assignment replay
over-merges (n_pred 30 << 140) - i.e. the default thresholds bracket the truth; `sweep.py`
exists to tune `match_thresh`/`new_thresh`/`k` toward n_pred ~ n_true (a documented tuning lever).
