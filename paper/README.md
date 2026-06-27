# Paper: "Precision at the Gate"

Title: Precision at the Gate - Per-Clip Detector Confidence Outranks Recall in
a Modernized DeepSORT Pipeline.

This directory is the research-paper track, separate from the course deliverable (the modernized
tracker and its report at the repository root).

## Thesis

On the six MOT-Challenge clips audited here, per-clip detector-confidence (precision) gating is a
larger and more reliable lever on HOTA than global recall maximization. Lowering the detector
confidence gate (admitting more low-confidence detections) lowers HOTA on all six videos, by an
amount strongly anti-correlated with per-clip detector precision (r = -0.91, n = 6); the input-size lever is
sequence-dependent. The modernized live pipeline drops the original DeepSORT min_confidence + NMS
gate, so ungated false positives spawn spurious tracks. Restoring the gate recovers the loss. The headline gain of the modernization decomposes as detection-dominated
(DetA much more than AssA), and, under perfect (ground-truth) detection, association quality is
nearly detector-bound: a generic ImageNet backbone nearly matches a dedicated REID model.

## Contributions

1. A detection-vs-association decomposition of the modernization gain.
2. A documented "more detections hurt" regime with a mechanistic explanation, shown across videos
   and anti-correlated with per-clip detector precision (r = -0.91, n = 6).
3. A gating-restoration ablation attributing the regression to the dropped min_confidence + NMS gate.
4. A practical per-clip precision-gating guideline.

## Experiments

- `experiments/run_gating_study.py` - per-video confidence sweep, an input-size sweep on the dense
  clips, and the gating-restoration ablation. Writes `experiments/gating_study.csv` (one row per
  configuration). Resilient: results are checkpointed and finished configurations are skipped on
  re-run.
- `experiments/analyze.py` - builds the figures and macros from `gating_study.csv`. The per-clip
  detector precision it plots is the pinned imgsz-1280 measurement (mirrored in
  `results/experiments.csv` as the `detector_imgsz1280` rows), not read from the per-run journal.

## Positioning note

The closest related work is ByteTrack, which argues for keeping low-confidence detections. This is a
different mechanism, not a contradiction: ByteTrack associates low-confidence boxes to existing
tracks, whereas the issue studied here is low-confidence boxes spawning new tracks when the gate is
absent. The paper must make this distinction explicit.
