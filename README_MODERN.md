# Modernized DeepSORT — Person Tracking with Pluggable Detectors & REID

A modernization of the original [nwojke/deep_sort](https://github.com/nwojke/deep_sort)
(this repo preserves its full commit history). The SORT core is kept intact; detection
and appearance are made **pluggable**, a **live per-frame pipeline** replaces the
precomputed-`.npy` flow, and a **standalone body-REID identity system** is added.

## Architecture

```
detectors/   pluggable detection (BaseDetector -> DetectionResult), lazy-import registry
reid/        pluggable appearance (BaseReIDExtractor -> L2-normed features), registry
pipeline/    live TrackingRunner (detect -> crop -> REID -> Detection -> SORT), live + gtbox
bodyreid/    persistent identity DB + kNN + window-vote + conflict resolution (Additional)
eval/        det P/R/F1, run_tracking, TrackEval/HOTA, reid-only HOTA, FPS, overlays
data/        MOT download/layout, GT-crop export
deep_sort/   ORIGINAL SORT core — unchanged except a feature-dimension guard in nn_matching
```

The single integration seam is the original contract `Detection(tlwh, confidence, feature)`
fed to `tracker.update(...)`. The cosine appearance metric is dimension-agnostic, so REID
models of any output size (128 / 512 / 2048) work unchanged.

## Model inventory (diversity → scoring)

| Type | Models | Sources |
|------|--------|---------|
| Detection | YOLOv8 (`yolo`), NanoDet (`nanodet`), MMDetection RTMDet (`mmdet`) | ultralytics, RangiLyu, open-mmlab (3 repos) |
| Segmentation | YOLOv8-seg (`yolo_seg`, mask→bbox) | ultralytics |
| REID | OSNet / OSNet-AIN / ResNet50 (`torchreid`), timm backbone (`timm`), mars-small128 (`mars`, baseline) | deep-person-reid, timm, original (≥2 sources) |
| REID (optional) | FastReID (`fastreid`) | JDAI-CV |

## Install

```bash
pip install -r requirements-modern.txt
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git   # torchreid (OSNet/ResNet50) — not on PyPI
pip install -U openmim && mim install mmengine "mmcv>=2.0" mmdet      # for the mmdet detector
git clone https://github.com/JonathonLuiten/TrackEval third_party/TrackEval && pip install -e third_party/TrackEval
```
See [data/README.md](data/README.md) for MOT data, NanoDet, and the mars baseline.

## Usage (run from the repo root)

```bash
python -m data.download_mot                                          # fetch + lay out MOT15/MOT16
python run.py --detector yolo --reid osnet_x1_0 --sequence MOT16-09  # single-sequence track
python -m eval.det_eval --detector yolo                              # detector P/R/F1 vs GT
python -m eval.run_tracking --detector yolo --reid osnet_x1_0        # tracking -> MOT results
python -m eval.trackeval_runner --benchmark MOT16 --trackers yolo__osnet_x1_0   # HOTA
python -m eval.reid_eval --reids osnet_x1_0 resnet50 mars            # REID-only HOTA (GT boxes)
python -m eval.fps_bench --detector yolo --reid osnet_x1_0 --sequence MOT16-09  # >=5 FPS check
python -m eval.run_tracking --detector yolo --reid osnet_x1_0 --bodyreid        # + identity system
python -m eval.make_overlays --detector yolo --reid osnet_x1_0 --sequence MOT16-09 --out overlays/best_MOT16-09.mp4
```

Per-video parameter overrides via `--override` (e.g. `--override tracker.max_cosine_distance=0.3`)
or per-sequence presets in `configs/sequences/<seq>.yaml`.

## Reproducing the report

The notebook [notebooks/DeepSORT_Modern.ipynb](notebooks/DeepSORT_Modern.ipynb) is a thin
orchestrator that runs install → download → baseline → detector/REID studies → best-combo
tracking → body-REID → overlays end-to-end in Colab. Numbers land in `results/experiments.csv`;
the writeup is in [report/report.md](report/report.md).

## Tests

```bash
py -3.14 tests/test_foundation.py     # config/contracts/nn_matching guard
py -3.14 tests/test_imports.py        # all adapters register
py -3.14 tests/test_pipeline_e2e.py   # full pipeline on a synthetic sequence
py -3.14 tests/test_eval_core.py      # IoU / P/R/F1 / journal
py -3.14 tests/test_bodyreid.py       # identity DB, vote, conflicts, fragmentation healing
```
