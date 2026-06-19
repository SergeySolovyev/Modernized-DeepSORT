# Modernized DeepSORT (HSE "Deep Learning for Computer Vision" final project)

This repository modernizes the original DeepSORT (`nwojke/deep_sort`, retained below and in the git
history) into a configurable multi-object tracker with selectable detectors and re-identification
(REID) models, a segmentation option, and a standalone body-REID identity component.

## Results (HOTA, TrackEval protocol, six MOT-Challenge videos)

| Configuration | mean HOTA | Delta vs baseline | per-video wins | FPS | real-time |
|---|---|---|---|---|---|
| Baseline (unmodified DeepSORT, provided detections + mars) | 40.17 | - | - | - | - |
| Modern (YOLOv8m + OSNet) | 52.54 | +12.37 | 6 / 6 | 9.87 | yes |

The modern configuration exceeds the unmodified baseline on each of the six videos while running in
real time on a Colab T4. Per-video numbers, the detector and REID studies, the body-REID clustering
metrics, and a parameter-tuning case study are given in the report.

## Repository contents

| Item | Path |
|---|---|
| Report (methods, experiments including negative results, results) | [report/report.md](report/report.md), [report/report.pdf](report/report.pdf) |
| Results table | [report/results_table.md](report/results_table.md) |
| Overlay videos (unmodified baseline and best configuration, MOT16-09) | [overlays/baseline_MOT16-09.mp4](overlays/baseline_MOT16-09.mp4), [overlays/best_MOT16-09.mp4](overlays/best_MOT16-09.mp4) |
| Colab notebook (install, data, baseline, studies, best configuration, overlays) | [notebooks/DeepSORT_Modern.ipynb](notebooks/DeepSORT_Modern.ipynb) |
| Additional project notes (architecture, how to run) | [README_MODERN.md](README_MODERN.md) |
| Detector / REID / body-REID / pipeline / evaluation packages | `detectors/`, `reid/`, `bodyreid/`, `pipeline/`, `eval/`, `configs/` |
| Experiment journal | `results/experiments.csv` |

## How to run

Open [notebooks/DeepSORT_Modern.ipynb](notebooks/DeepSORT_Modern.ipynb) in Google Colab
(Runtime -> GPU) and run all cells. The notebook clones this repository, installs dependencies,
downloads MOT15 and MOT16, reproduces the baseline and the modern pipeline, and renders the overlay
videos. The SORT core (`deep_sort/`) is unchanged; the modern components are added behind the
`BaseDetector` and `BaseReIDExtractor` interfaces.

The original upstream DeepSORT README is retained below for attribution and reproducibility.

---

# Deep SORT (original upstream README)

## Introduction

This repository contains code for *Simple Online and Realtime Tracking with a Deep Association Metric* (Deep SORT).
We extend the original [SORT](https://github.com/abewley/sort) algorithm to
integrate appearance information based on a deep appearance descriptor.
See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.

## Installation

First, clone the repository and install dependencies:
```
git clone https://github.com/nwojke/deep_sort.git
cd deep_sort

# The following command installs all the dependencies required to run the
# tracker and regenerate detections. If you only need to run the tracker with
# existing detections, you can use pip install -r requirements.txt instead.
pip install -r requirements-gpu.txt
```
Then, download pre-generated detections and the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).

*NOTE:* The candidate object locations of our pre-generated detections are
taken from the following paper:
```
F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, J. Yan. POI: Multiple Object Tracking with
High Performance Detection and Appearance Feature. In BMTT, SenseTime Group
Limited, 2016.
```
We have replaced the appearance descriptor with a custom deep convolutional
neural network (see below).

## Running the tracker

The following example starts the tracker on one of the
[MOT16 benchmark](https://motchallenge.net/data/MOT16/)
sequences.
We assume resources have been extracted to the repository root directory and
the MOT16 benchmark data is in `./MOT16`:
```
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
```
Check `python deep_sort_app.py -h` for an overview of available options.
There are also scripts in the repository to visualize results, generate videos,
and evaluate the MOT challenge benchmark.

## Generating detections

Beside the main tracking application, this repository contains a script to
generate features for person re-identification, suitable to compare the visual
appearance of pedestrian bounding boxes using cosine similarity.
The following example generates these features from standard MOT challenge
detections. Again, we assume resources have been extracted to the repository
root directory and MOT16 data is in `./MOT16`:
```
python tools/generate_detections.py \
    --model=resources/networks/mars-small128.pb \
    --mot_dir=./MOT16/train \
    --output_dir=./resources/detections/MOT16_train
```
The model has been generated with TensorFlow 1.5. If you run into
incompatibility, re-export the frozen inference graph to obtain a new
`mars-small128.pb` that is compatible with your version:
```
python tools/freeze_model.py
```
The ``generate_detections.py`` stores for each sequence of the MOT16 dataset
a separate binary file in NumPy native format. Each file contains an array of
shape `Nx138`, where N is the number of detections in the corresponding MOT
sequence. The first 10 columns of this array contain the raw MOT detection
copied over from the input file. The remaining 128 columns store the appearance
descriptor. The files generated by this command can be used as input for the
`deep_sort_app.py`.

**NOTE**: If ``python tools/generate_detections.py`` raises a TensorFlow error,
try passing an absolute path to the ``--model`` argument. This might help in
some cases.

## Training the model

To train the deep association metric model we used a novel [cosine metric learning](https://github.com/nwojke/cosine_metric_learning) approach which is provided as a separate repository.

## Highlevel overview of source files

In the top-level directory are executable scripts to execute, evaluate, and
visualize the tracker. The main entry point is in `deep_sort_app.py`.
This file runs the tracker on a MOTChallenge sequence.

In package `deep_sort` is the main tracking code:

* `detection.py`: Detection base class.
* `kalman_filter.py`: A Kalman filter implementation and concrete
   parametrization for image space filtering.
* `linear_assignment.py`: This module contains code for min cost matching and
   the matching cascade.
* `iou_matching.py`: This module contains the IOU matching metric.
* `nn_matching.py`: A module for a nearest neighbor matching metric.
* `track.py`: The track class contains single-target track data such as Kalman
  state, number of hits, misses, hit streak, associated feature vectors, etc.
* `tracker.py`: This is the multi-target tracker class.

The `deep_sort_app.py` expects detections in a custom format, stored in .npy
files. These can be computed from MOTChallenge detections using
`generate_detections.py`. We also provide
[pre-generated detections](https://drive.google.com/open?id=1VVqtL0klSUvLnmBKS89il1EKC3IxUBVK).

## Citing DeepSORT

If you find this repo useful in your research, please consider citing the following papers:

    @inproceedings{Wojke2017simple,
      title={Simple Online and Realtime Tracking with a Deep Association Metric},
      author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
      booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
      year={2017},
      pages={3645--3649},
      organization={IEEE},
      doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
      title={Deep Cosine Metric Learning for Person Re-identification},
      author={Wojke, Nicolai and Bewley, Alex},
      booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
      year={2018},
      pages={748--756},
      organization={IEEE},
      doi={10.1109/WACV.2018.00087}
    }
