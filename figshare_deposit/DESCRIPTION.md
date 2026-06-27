# Figshare deposit — ready-to-paste metadata

Copy each field into the matching box on figshare.com when you create the item.

---

## Title
Precision at the Gate: Per-Clip Detector Confidence Outranks Recall in a Modernized DeepSORT Pipeline

## Item type
Preprint

## Authors
Sergei Solovev — HSE University
(add your ORCID in the author box if you have one)

## Categories  (pick the closest from Figshare's list)
- Computer Vision and Multimedia Computation
- Pattern Recognition
- Machine Learning
- Artificial Intelligence

## Keywords
multi-object tracking; tracking-by-detection; DeepSORT; HOTA; detector confidence;
non-maximum suppression; gating; person re-identification; MOT-Challenge; YOLOv8; OSNet

## License
CC BY 4.0   (recommended default — free reuse with attribution)

## Description / Abstract
Modernizing a 2017-era DeepSORT tracker by swapping its precomputed detections and fixed
appearance encoder for a contemporary detector (YOLOv8m) and re-identification model (OSNet)
raises mean HOTA on six MOT-Challenge videos from 40.17 to 52.63. This headline gain is
unsurprising; the contribution of this paper is the audit underneath it.

The gain is detection-dominated: on the MOT16 split the detection term rises by +12.30 HOTA
against only +6.25 for association, and under ground-truth boxes a generic ImageNet backbone
reaches 89.2 HOTA versus 89.9 for a dedicated OSNet. The mean also hides a regression: at its
default gate the modernized tracker scores below the 2017 baseline on the least-precise clip
(TUD-Campus, 35.14 vs 39.86 HOTA). Across all six videos the lowest detector-confidence gate
scores no higher than the highest, and on the low-precision clips lowering the gate sharply
lowers HOTA, with the size of the penalty tracking per-clip detector precision (Pearson
r = -0.91, n = 6; reported as a hypothesis-generating association, not a calibrated law).

The cause is specific and reproducible: the modern live pipeline drops the original DeepSORT
confidence and non-maximum-suppression admission gate, so ungated false positives spawn spurious
tracks. A gating-restoration ablation recovers the loss, and raising the per-clip confidence
threshold lifts the worst sequence from a 4.72-point deficit to +6.43 over baseline. We frame
the result as an audit rather than a new method: modern SORT-family trackers already instantiate
this admission gate as a tuned hyperparameter, and a routine modernization can silently drop it.
All detector results use a pinned configuration (ultralytics 8.4.79, yolov8m.pt, input size 1280).

Code, data, and reproduction scripts: https://github.com/SergeySolovyev/Modernized-DeepSORT

## References / Related materials  (add as links in Figshare)
- https://github.com/SergeySolovyev/Modernized-DeepSORT

## Funding
(none / leave blank, or add HSE University if applicable)
