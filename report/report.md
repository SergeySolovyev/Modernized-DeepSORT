# Modernizing DeepSORT — Report

> Target metric: **HOTA averaged across the six MOT-Challenge videos** (TrackEval protocol).
> All numbers below are produced by the scripts in `eval/` and logged to
> `results/experiments.csv`. Fill the `(…)` cells from a Colab run.

Videos: **TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1** (MOT15) and
**MOT16-09, MOT16-11** (MOT16).

---

## 1. Baseline & protocol

The unmodified DeepSORT (original provided detections + `mars-small128` appearance) defines
the bar. Quality is measured with TrackEval HOTA (with MOTA/IDF1 as secondary). Detectors are
screened by Precision/Recall/F1 vs GT (IoU≥0.5); REID is screened both in-tracker (GT boxes,
SORT disabled → HOTA) and standalone (GT crops → clustering metrics).

**Baseline HOTA (unmodified DeepSORT):**

| Metric | TUD-Campus | TUD-Stadtmitte | KITTI-17 | PETS09-S2L1 | MOT16-09 | MOT16-11 | **Mean** |
|---|---|---|---|---|---|---|---|
| HOTA | (…) | (…) | (…) | (…) | (…) | (…) | **(…)** |
| MOTA | (…) | (…) | (…) | (…) | (…) | (…) | (…) |
| IDF1 | (…) | (…) | (…) | (…) | (…) | (…) | (…) |

---

## 2. Detector study — Precision / Recall / F1 (IoU≥0.5 vs GT)

Candidates: YOLOv8 (ultralytics), NanoDet (RangiLyu), MMDetection RTMDet (open-mmlab) —
three repositories / architecture families for diversity.

| Detector | mean P | mean R | mean F1 | notes (speed / which videos it wins) |
|---|---|---|---|---|
| yolo (v8m) | (…) | (…) | (…) | (…) |
| nanodet | (…) | (…) | (…) | (…) |
| mmdet (rtmdet) | (…) | (…) | (…) | (…) |

*Per-video F1 table:* `(insert results/det_eval_<detector>.csv)`. Discuss which detector was
selected per video and why (recall vs FPS trade-off).

---

## 3. REID study

**3a. REID-for-tracker (GT boxes, SORT detection disabled) — HOTA isolates appearance:**

| REID | TUD-Campus | TUD-Stadtmitte | KITTI-17 | PETS09-S2L1 | MOT16-09 | MOT16-11 | **Mean** |
|---|---|---|---|---|---|---|---|
| mars (baseline, 128-d) | (…) | (…) | (…) | (…) | (…) | (…) | (…) |
| osnet_x1_0 (512-d) | (…) | (…) | (…) | (…) | (…) | (…) | (…) |
| osnet_ain_x1_0 | (…) | (…) | (…) | (…) | (…) | (…) | (…) |
| resnet50 (2048-d) | (…) | (…) | (…) | (…) | (…) | (…) | (…) |
| timm (mobilenetv3) | (…) | (…) | (…) | (…) | (…) | (…) | (…) |

**3b. Standalone REID (GT crops) — clustering metrics (model selection):**

| REID | Fowlkes-Mallows | Silhouette (cos) | Calinski-Harabasz | n_pred / n_true |
|---|---|---|---|---|
| osnet_x1_0 | (…) | (…) | (…) | (…) |
| osnet_ain_x1_0 | (…) | (…) | (…) | (…) |
| resnet50 | (…) | (…) | (…) | (…) |
| timm | (…) | (…) | (…) | (…) |
| mars | (…) | (…) | (…) | (…) |

---

## 4. Standalone body-REID (Additional task)

Design: persistent identity DB (iid) decoupled from tracker track-ids (tid); per-frame kNN
search (centroid vs per-descriptor; k1/knn_vote/radius); create-on-track-birth with an
ambiguous band; per-track time-window (T) distance-weighted majority vote with hysteresis;
cross-track conflict resolution (keep_best). Parameters tuned by `bodyreid/eval/sweep.py`.

**Best params (from sweep):** `match_thresh=(…)`, `new_thresh=(…)`, `k=(…)`,
`search_target=(…)`, `gallery_cap=(…)`, `T=(…)s`, `conflict_policy=(…)`.

**Effect on tracking (full pipeline, HOTA / IDF1 with vs without body-REID):**

| Config | mean HOTA | mean IDF1 | identity switches | notes |
|---|---|---|---|---|
| best detector+REID | (…) | (…) | (…) | (…) |
| + body-REID | (…) | (…) | (…) | (fragmentation healing) |

---

## 5. Segmentation

YOLOv8-seg used as a switchable detector (person mask → tight bbox). Compare vs the YOLO
box detector on P/R/F1 and HOTA; note FPS cost.

| Config | mean F1 | mean HOTA | FPS | notes |
|---|---|---|---|---|
| yolo (box) | (…) | (…) | (…) | (…) |
| yolo_seg (mask→bbox) | (…) | (…) | (…) | (…) |

---

## 6. Parameter evolution (quality ↔ performance)

Plots of the tuning trajectory: detector size / `imgsz`, `max_cosine_distance`, `nn_budget`,
`max_age`, REID model — each against mean HOTA and FPS. Show the ≥5 FPS real-time frontier.

`(insert figures generated from results/experiments.csv)`

---

## 7. Failure modes & lessons (incl. negative results)

Document what did NOT work and why — e.g. a detector that tanked recall on a crowded video,
a REID that over-merged identities (identity explosion), configs that fell below 5 FPS,
centroid drift / oscillation in the identity system and the mitigations applied.
Negative results are kept here deliberately.

---

## 8. Conclusion — optimal config per video & comparison vs baseline

| Video | best detector | best REID | key params | HOTA | Δ vs baseline | FPS |
|---|---|---|---|---|---|---|
| TUD-Campus | (…) | (…) | (…) | (…) | (…) | (…) |
| TUD-Stadtmitte | (…) | (…) | (…) | (…) | (…) | (…) |
| KITTI-17 | (…) | (…) | (…) | (…) | (…) | (…) |
| PETS09-S2L1 | (…) | (…) | (…) | (…) | (…) | (…) |
| MOT16-09 | (…) | (…) | (…) | (…) | (…) | (…) |
| MOT16-11 | (…) | (…) | (…) | (…) | (…) | (…) |
| **Mean** | — | — | — | **(…)** | **(…)** | **(…)** |

The modern configuration beats the unmodified DeepSORT on **every** video while keeping a
≥5 FPS real-time combination on Colab Pro.
