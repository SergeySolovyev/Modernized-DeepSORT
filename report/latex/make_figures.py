"""Generate the report figures (matplotlib) from the verified results.

All values are taken from report/results_table.md and report/report.md (the live
Colab run, 2026-06-19). Figures are written as PNG to images/outputs/ next to this
script and are included by main.tex.

  python report/latex/make_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
    "axes.axisbelow": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.8,
    "legend.fontsize": 10, "font.family": "DejaVu Sans",
})

SLATE, BLUE, ORANGE = "#5B6B7B", "#3B6FA0", "#E8730C"
LBLUE, LORANGE, GREEN, GRAY = "#A8CCE0", "#F4A259", "#2E8B57", "#9AA3AD"
EDGE = "#333333"


def gridy(ax):
    ax.grid(axis="y"); ax.set_axisbelow(True)


def vlabels(ax, bars, fmt="%.2f", fs=8, dy=2):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt % h, (b.get_x() + b.get_width() / 2, h), xytext=(0, dy),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig); print("wrote", name)


# 1) HOTA per video: baseline vs two modern configurations
def fig_hota_per_video():
    seqs = ["TUD-Campus", "TUD-Stadt.", "KITTI-17", "PETS09-S2L1", "MOT16-09", "MOT16-11", "Mean"]
    base = [39.86, 36.75, 43.41, 44.84, 36.24, 39.95, 40.17]
    timm = [37.57, 57.63, 45.82, 50.89, 45.56, 48.73, 47.70]
    osn = [46.98, 59.62, 48.75, 60.28, 48.57, 51.07, 52.54]
    x = np.arange(len(seqs)); w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5.2))
    b1 = ax.bar(x - w, base, w, label="Baseline (provided det + mars)", color=SLATE, edgecolor=EDGE, linewidth=0.5)
    b2 = ax.bar(x, timm, w, label="YOLOv8m + timm", color=BLUE, edgecolor=EDGE, linewidth=0.5)
    b3 = ax.bar(x + w, osn, w, label="YOLOv8m + OSNet", color=ORANGE, edgecolor=EDGE, linewidth=0.5)
    ax.set_ylabel("HOTA"); ax.set_ylim(0, 72)
    ax.set_title("HOTA per video: baseline vs modern configurations")
    ax.set_xticks(x); ax.set_xticklabels(seqs, rotation=18, ha="right")
    ax.axvline(5.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
    gridy(ax); ax.legend(loc="upper left", ncol=3)
    for b in (b1, b2, b3): vlabels(ax, b, fs=7)
    save(fig, "fig_hota_per_video.png")


# 2) REID-only HOTA (GT boxes; isolates appearance)
def fig_reid_only():
    names = ["OSNet", "timm", "OSNet-AIN", "OSNet-x0.25"]
    vals = [89.93, 89.22, 87.33, 86.13]
    cols = [ORANGE, BLUE, LORANGE, LBLUE]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b = ax.bar(names, vals, color=cols, edgecolor=EDGE, linewidth=0.5, width=0.6)
    ax.set_ylabel("mean HOTA"); ax.set_ylim(80, 92)
    ax.set_title("REID-only HOTA with ground-truth boxes (mean over six videos)")
    gridy(ax); vlabels(ax, b, fmt="%.2f", fs=10)
    save(fig, "fig_reid_only.png")


# 3) Detector precision / recall / F1
def fig_detector_prf():
    metrics = ["Precision", "Recall", "F1"]
    yolo = [0.683, 0.831, 0.740]
    yseg = [0.696, 0.831, 0.749]
    x = np.arange(len(metrics)); w = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.6))
    b1 = ax.bar(x - w / 2, yolo, w, label="YOLOv8m (box)", color=BLUE, edgecolor=EDGE, linewidth=0.5)
    b2 = ax.bar(x + w / 2, yseg, w, label="YOLOv8m-seg (mask->bbox)", color=ORANGE, edgecolor=EDGE, linewidth=0.5)
    ax.set_ylabel("score (IoU >= 0.5)"); ax.set_ylim(0, 1.0)
    ax.set_title("Detector evaluation vs ground truth (mean over six videos)")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    gridy(ax); ax.legend(loc="lower right")
    for b in (b1, b2): vlabels(ax, b, fmt="%.3f", fs=9)
    save(fig, "fig_detector_prf.png")


# 4) TUD-Campus tuning sweep (horizontal; baseline reference)
def fig_tuning_sweep():
    cand = [("conf 0.45 (selected)", 46.98), ("conf 0.35", 44.61), ("nms0.7 + conf0.35", 42.49),
            ("min_confidence 0.30", 42.22), ("nms_max_overlap 0.5", 41.96),
            ("nms0.7 + minconf + h15", 40.39), ("min_height 20", 39.85),
            ("control (conf 0.25)", 39.65), ("nms0.7 alone", 38.40),
            ("imgsz 1280", 35.03), ("imgsz 1536", 29.75)]
    cand.sort(key=lambda kv: kv[1])
    names = [c[0] for c in cand]; vals = [c[1] for c in cand]
    base = 39.86
    cols = [GREEN if v > base else GRAY for v in vals]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, edgecolor=EDGE, linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.axvline(base, color="#C0392B", linewidth=1.4, linestyle="--", label="baseline HOTA = 39.86")
    ax.set_xlabel("TUD-Campus HOTA"); ax.set_xlim(0, 52)
    ax.set_title("TUD-Campus per-video tuning sweep")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    for yi, v in zip(y, vals):
        ax.annotate("%.2f" % v, (v, yi), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8)
    ax.legend(loc="lower right")
    save(fig, "fig_tuning_sweep.png")


# 5) Metric decomposition on the combined MOT16 split
def fig_metric_decomposition():
    metrics = ["HOTA", "MOTA", "IDF1", "DetA", "AssA"]
    base = [38.65, 46.34, 50.65, 37.85, 39.52]
    mod = [47.68, 46.75, 52.18, 50.15, 45.77]
    x = np.arange(len(metrics)); w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    b1 = ax.bar(x - w / 2, base, w, label="Baseline (provided det + mars)", color=SLATE, edgecolor=EDGE, linewidth=0.5)
    b2 = ax.bar(x + w / 2, mod, w, label="Modern (YOLOv8m + timm)", color=ORANGE, edgecolor=EDGE, linewidth=0.5)
    ax.set_ylabel("score"); ax.set_ylim(0, 62)
    ax.set_title("Metric decomposition on the combined MOT16 split")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    gridy(ax); ax.legend(loc="upper left", ncol=2)
    for b in (b1, b2): vlabels(ax, b, fmt="%.2f", fs=8)
    save(fig, "fig_metric_decomposition.png")


# 6) Segmentation: detection F1 and throughput
def fig_segmentation():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    names = ["YOLOv8m\n(box)", "YOLOv8m-seg\n(mask->bbox)"]
    f1 = [0.740, 0.749]; fps = [14.68, 8.91]
    b1 = a1.bar(names, f1, color=[BLUE, ORANGE], edgecolor=EDGE, linewidth=0.5, width=0.55)
    a1.set_ylabel("mean detection F1"); a1.set_ylim(0, 0.9); gridy(a1)
    a1.set_title("Detection quality"); vlabels(a1, b1, fmt="%.3f", fs=10)
    b2 = a2.bar(names, fps, color=[BLUE, ORANGE], edgecolor=EDGE, linewidth=0.5, width=0.55)
    a2.axhline(5, color="#C0392B", linewidth=1.2, linestyle="--", label="real-time bar (5 FPS)")
    a2.set_ylabel("mean FPS"); a2.set_ylim(0, 18); gridy(a2)
    a2.set_title("Throughput"); a2.legend(loc="upper right"); vlabels(a2, b2, fmt="%.2f", fs=10)
    fig.suptitle("Box detector vs segmentation detector (same OSNet REID)", fontsize=14)
    save(fig, "fig_segmentation.png")


# 7) Standalone body-REID clustering (predicted vs true identity count)
def fig_bodyreid_cluster():
    modes = ["Embedding\n(Agglomerative)", "Assignment\n(headless DB replay)"]
    npred = [420, 30]; fmi = [0.630, 0.242]
    x = np.arange(len(modes)); w = 0.5
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b = ax.bar(x, npred, w, color=[BLUE, ORANGE], edgecolor=EDGE, linewidth=0.5)
    ax.axhline(140, color="#C0392B", linewidth=1.4, linestyle="--", label="true identities = 140")
    ax.set_xticks(x); ax.set_xticklabels(modes)
    ax.set_ylabel("predicted identity count"); ax.set_ylim(0, 470)
    ax.set_title("Standalone body-REID: predicted vs true identity count (21,105 GT crops)")
    gridy(ax); ax.legend(loc="upper right")
    for xi, n, f in zip(x, npred, fmi):
        ax.annotate("n_pred = %d\nFowlkes-Mallows = %.3f" % (n, f), (xi, n), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    save(fig, "fig_bodyreid_cluster.png")


# 8) Per-frame latency by stage
def fig_stage_latency():
    stages = ["Detector", "REID", "Tracker"]
    ms = [32.4, 32.7, 20.5]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b = ax.bar(stages, ms, color=[BLUE, ORANGE, SLATE], edgecolor=EDGE, linewidth=0.5, width=0.55)
    ax.set_ylabel("per-frame time (ms)"); ax.set_ylim(0, 40)
    ax.set_title("Per-frame latency by stage (MOT16-09, YOLOv8m + OSNet, T4)")
    gridy(ax); vlabels(ax, b, fmt="%.1f ms", fs=10)
    ax.annotate("overall throughput: 9.87 FPS (real-time)", (0.5, 0.92),
                xycoords="axes fraction", ha="center", fontsize=10, color="#333333")
    save(fig, "fig_stage_latency.png")


# 9) Pipeline / architecture diagram
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, text, fc, tc="#111111", fs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    linewidth=1.0, edgecolor=EDGE, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     linewidth=1.1, color="#555555"))

    y = 17; h = 7
    box(1, y, 13, h, "Frame\n(MOT image)", "#EEEEEE")
    box(17, y, 15, h, "Detector\n(YOLOv8m / -seg)", LORANGE)
    box(35, y, 12, h, "Crop +\nREID (OSNet)", LBLUE)
    box(50, y, 16, h, "Detection\n(tlwh, conf, feature)", "#EEEEEE")
    box(69, y, 17, h, "SORT core (unchanged):\nKalman + Hungarian +\ncosine gating", "#DDE7F0", fs=9)
    box(89, y, 10, h, "Tracks", "#EEEEEE")
    for x1, x2 in [(14, 17), (32, 35), (47, 50), (66, 69), (86, 89)]:
        arrow(x1, y + h / 2, x2, y + h / 2)
    # body-REID branch
    box(69, 3, 17, 8, "Standalone body-REID:\nidentity DB + kNN +\nwindow vote + conflicts", "#FBE3CF", fs=9)
    arrow(94, y, 94, 11)
    ax.text(95.5, 14, "per-frame\nhook", ha="left", va="center", fontsize=8, color="#777777")
    ax.text(50, 28, "Pluggable detector and REID adapters feed the unchanged SORT core; "
                    "the body-REID component reads tracker output without modifying it.",
            ha="center", va="center", fontsize=9, color="#333333")
    save(fig, "fig_pipeline.png")


if __name__ == "__main__":
    fig_pipeline()
    fig_hota_per_video()
    fig_reid_only()
    fig_detector_prf()
    fig_tuning_sweep()
    fig_metric_decomposition()
    fig_segmentation()
    fig_bodyreid_cluster()
    fig_stage_latency()
    print("all figures written to", OUT)
