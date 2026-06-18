"""Generate notebooks/DeepSORT_Modern.ipynb (thin orchestrator) as valid nbformat-4 JSON.

Run: py -3.14 notebooks/_build_notebook.py
Keeping logic in the scripts (eval/, data/, bodyreid/) and only orchestration in the
notebook preserves the "organized scripts, not notebooks" scoring point.
"""
import json
import os

DET, REID = "yolo", "osnet_x1_0"   # default best combo (tune per video)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in text.split("\n")]}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in text.split("\n")]}


cells = [
    md("# Modernized DeepSORT — execution notebook\n\n"
       "Thin orchestrator: every step calls the repo's scripts (logic lives in `eval/`, "
       "`data/`, `bodyreid/`). Runs top-to-bottom on **Colab Pro (GPU)**.\n\n"
       "**Instructions:** set `REPO_URL`, Runtime → GPU, then Run all. Results are logged to "
       "`results/experiments.csv`; overlays to `overlays/`."),

    code("# 1) Clone the repo and enter it (public repo -> no auth; if private, use a PAT URL)\n"
         "REPO_URL = 'https://github.com/SergeySolovyev/Modernized-DeepSORT.git'\n"
         "import os\n"
         "if not os.path.isdir('Modernized-DeepSORT'):\n"
         "    !git clone $REPO_URL Modernized-DeepSORT\n"
         "%cd Modernized-DeepSORT"),

    code("# 2) Install dependencies\n"
         "!pip -q install -r requirements-modern.txt\n"
         "# torchreid (OSNet/ResNet50) is not on PyPI under this name -> install from source:\n"
         "!pip -q install git+https://github.com/KaiyangZhou/deep-person-reid.git\n"
         "!pip -q install -U openmim && mim install mmengine 'mmcv>=2.0' mmdet\n"
         "!git clone -q https://github.com/JonathonLuiten/TrackEval third_party/TrackEval || true\n"
         "!pip -q install -e third_party/TrackEval"),

    code("# 3) GPU check\n!nvidia-smi"),

    code("# 4) Download + lay out MOT15/MOT16 (TrackEval structure)\n"
         "!python -m data.download_mot"),

    md("## Baseline — unmodified DeepSORT (provided detections + mars-small128)\n"
       "Faithful baseline via the original legacy path. Requires `mars-small128.pb` at "
       "`third_party/deep_sort_data/` (see data/README.md)."),
    code("# 5) Baseline: unmodified DeepSORT (provided detections + mars-small128).\n"
         "# Requires third_party/deep_sort_data/mars-small128.pb (see data/README.md).\n"
         "!python -m eval.run_baseline --mars third_party/deep_sort_data/mars-small128.pb\n"
         "!python -m eval.trackeval_runner --benchmark MOT15 --trackers baseline\n"
         "!python -m eval.trackeval_runner --benchmark MOT16 --trackers baseline"),

    md("## Detector study — Precision / Recall / F1 vs GT (IoU≥0.5)"),
    code("for det in ['yolo', 'nanodet', 'mmdet']:\n"
         "    !python -m eval.det_eval --detector $det --device cuda"),

    md("## REID study — REID-only HOTA (GT boxes, SORT detection disabled)"),
    code("!python -m eval.reid_eval --reids osnet_x1_0 osnet_ain_x1_0 resnet50 timm_mobilenet mars --device cuda"),

    md("## Full pipeline — best combo, live tracking → HOTA"),
    code("DET, REID = '%s', '%s'\n" % (DET, REID) +
         "!python -m eval.run_tracking --detector $DET --reid $REID --device cuda\n"
         "!python -m eval.trackeval_runner --benchmark MOT15 --trackers ${DET}__${REID}\n"
         "!python -m eval.trackeval_runner --benchmark MOT16 --trackers ${DET}__${REID}"),

    code("# FPS (must be >= 5 FPS for the real-time requirement)\n"
         "!python -m eval.fps_bench --detector $DET --reid $REID --sequence MOT16-09 --device cuda"),

    md("## Additional task — standalone body-REID identity system"),
    code("# Standalone REID model/param selection on GT crops\n"
         "!python -m data.prepare_gt_crops\n"
         "!python -m bodyreid.eval.extract_gt --reid $REID --out descriptors_$REID.npz --device cuda\n"
         "!python -m bodyreid.eval.cluster_eval --npz descriptors_$REID.npz\n"
         "!python -m bodyreid.eval.sweep --npz descriptors_$REID.npz --reid $REID"),
    code("# Full pipeline WITH the identity system\n"
         "!python -m eval.run_tracking --detector $DET --reid $REID --bodyreid --device cuda"),

    md("## Segmentation — YOLOv8-seg (mask→bbox)"),
    code("!python -m eval.det_eval --detector yolo_seg --device cuda\n"
         "!python -m eval.run_tracking --detector yolo_seg --reid $REID --device cuda"),

    md("## Overlays — baseline vs best"),
    code("import os; os.makedirs('overlays', exist_ok=True)\n"
         "!python -m eval.make_overlays --detector gt   --reid mars  --sequence MOT16-09 --mode gtbox --out overlays/baseline_MOT16-09.mp4 --device cuda\n"
         "!python -m eval.make_overlays --detector $DET --reid $REID --sequence MOT16-09 --out overlays/best_MOT16-09.mp4 --device cuda"),

    md("## Results summary"),
    code("# Auto-build the per-video HOTA table (tracker x video + Mean, Delta vs baseline)\n"
         "!python -m eval.summarize\n"
         "import pandas as pd\n"
         "print(open('report/results_table.md').read())\n"
         "df = pd.read_csv('results/experiments.csv')\n"
         "df.tail(40)"),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepSORT_Modern.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print("wrote", out, "(%d cells)" % len(cells))
