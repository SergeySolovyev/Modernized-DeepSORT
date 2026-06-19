# Results — live Colab run (Colab Pro, Tesla T4, 2026-06-19)

All six MOT-Challenge train sequences. HOTA via TrackEval (numpy-2.x patched). Trackers:
`baseline` = unmodified DeepSORT (provided detections + mars-small128); `yolo__*` = modern
live pipeline (YOLOv8m detector + REID); `gt__*__gtbox` = REID-only study (GT boxes, SORT
detection disabled — isolates the appearance model).

## HOTA per video

| tracker | TUD-Campus | TUD-Stadtmitte | KITTI-17 | PETS09-S2L1 | MOT16-09 | MOT16-11 | **Mean** | Δ vs baseline |
|---|---|---|---|---|---|---|---|---|
| baseline (unmodified) | 39.86 | 36.75 | 43.41 | 44.84 | 36.24 | 39.95 | **40.17** | — |
| **yolo__osnet** (best live) | 39.65 | 59.62 | 48.75 | 60.28 | 48.57 | 51.07 | **51.32** | **+11.15** |
| yolo__timm_mobilenet | 37.57 | 57.63 | 45.82 | 50.89 | 45.56 | 48.73 | **47.70** | +7.53 |
| gt__osnet__gtbox (REID-only) | 86.67 | 94.11 | 82.98 | 88.52 | 93.46 | 93.82 | **89.93** | +49.76 |
| gt__timm_mobilenet__gtbox | 86.67 | 94.11 | 83.89 | 91.71 | 85.38 | 93.59 | **89.22** | +49.05 |
| gt__osnet_ain__gtbox | 86.67 | 87.30 | 80.87 | 90.28 | 85.44 | 93.45 | **87.33** | +47.16 |
| gt__osnet_fast__gtbox | 86.67 | 87.30 | 78.58 | 83.86 | 87.33 | 93.04 | **86.13** | +45.96 |

**Headline:** the modernized tracker (**YOLOv8 + OSNet**, OSNet via boxmot) reaches **mean HOTA
51.32 vs 40.17** for unmodified DeepSORT (**+11.15**), beating the baseline on **5 of 6**
videos. **TUD-Campus** is a near-tie just below (39.65 vs 39.86) — a 71-frame dense clip that
needs per-video tuning (`configs/sequences/TUD-Campus.yaml`: lower `conf`, adjust `max_age`).

## Real-time (FPS, MOT16-09, T4, warmup-excluded)

YOLOv8 + OSNet: **overall 9.87 FPS** (det 30.9 / reid 30.6 / track 48.7) → **REAL-TIME (≥5 FPS): YES**.

## REID-only study (GT boxes → HOTA isolates appearance)

OSNet is the strongest appearance model (mean HOTA 89.93), narrowly ahead of timm (89.22),
then OSNet-AIN (87.33) and OSNet-x0.25 (86.13). High absolute values are expected here because
GT boxes make detection near-perfect, so HOTA mostly reflects association quality.

## Standalone body-REID (OSNet descriptors: 21,105 GT crops, 140 true identities)

| mode | Fowlkes-Mallows | Silhouette (cos) | Calinski-Harabasz | n_pred / n_true |
|---|---|---|---|---|
| embedding (Agglomerative @ match_thresh) | 0.630 | 0.371 | 98.3 | 420 / 140 |
| assignment (headless IdentityDatabase replay) | 0.242 | — | — | 30 / 140 |

The embedding clustering over-splits (n_pred 420 ≫ 140) while the live-assignment replay
over-merges (n_pred 30 ≪ 140) — i.e. the default thresholds bracket the truth; `sweep.py`
exists to tune `match_thresh`/`new_thresh`/`k` toward n_pred ≈ n_true (a documented tuning lever).
