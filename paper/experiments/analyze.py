"""Build the paper figures and the numeric macros from the gating study.

Reads paper/experiments/gating_study.csv (the confidence/imgsz/ablation sweep), writes the PDF
figures to paper/figures/, prints the macro values, and updates nHurt / corrPearson /
ablationRecover in paper/sections/results_macros.tex. Per-clip detector precision is the pinned
imgsz-1280 measurement carried in PREC0 below (see the Reproducibility note in the paper).

  python paper/experiments/analyze.py        # run from the repo root
"""
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.normpath(os.path.join(HERE, "..", "figures"))
MACROS = os.path.normpath(os.path.join(HERE, "..", "sections", "results_macros.tex"))
GATING = os.path.join(HERE, "gating_study.csv")
JOURNAL = os.path.join("results", "experiments.csv")
os.makedirs(FIG, exist_ok=True)

SEQS = ["TUD-Campus", "TUD-Stadtmitte", "KITTI-17", "PETS09-S2L1", "MOT16-09", "MOT16-11"]
SHORT = {"TUD-Campus": "TUD-Campus", "TUD-Stadtmitte": "TUD-Stadt.", "KITTI-17": "KITTI-17",
         "PETS09-S2L1": "PETS09", "MOT16-09": "MOT16-09", "MOT16-11": "MOT16-11"}
DENSE = ["TUD-Campus", "KITTI-17"]
# Per-clip detector precision at conf 0.25, imgsz 1280 (the gating-study config),
# measured with the pinned weights: ultralytics 8.4.79, yolov8m.pt
# sha256 5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5.
PREC0 = {"TUD-Campus": 0.432, "TUD-Stadtmitte": 0.892, "KITTI-17": 0.480,
         "PETS09-S2L1": 0.850, "MOT16-09": 0.623, "MOT16-11": 0.623}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "font.family": "serif", "axes.grid": True, "axes.axisbelow": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "legend.fontsize": 7.5,
})
BLUE, ORANGE, GRAY, GREEN, RED = "#3B6FA0", "#E8730C", "#5B6B7B", "#2E8B57", "#C0392B"


def load_gating():
    g = {}
    if os.path.exists(GATING):
        for r in csv.DictReader(open(GATING, encoding="utf-8")):
            try:
                g[(r["seq"], r["label"])] = float(r["HOTA"])
            except (ValueError, KeyError):
                pass
    return g


def load_precision():
    # Use the pinned imgsz-1280 precision (PREC0). We deliberately do NOT read
    # results/experiments.csv here: that journal records the report track's
    # default-imgsz detector run, whose precision differs from the gating study's
    # imgsz-1280 configuration and would mix two settings on the money plot.
    return dict(PREC0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name)); plt.close(fig); print("wrote", name)


def fig_hota_vs_conf(g):
    confs = [0.15, 0.25, 0.35, 0.45]
    labels = ["conf%03d" % int(round(c * 100)) for c in confs]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for seq in SEQS:
        ys = [g.get((seq, lb), np.nan) for lb in labels]
        sty = "-o" if seq in DENSE else "--s"
        ax.plot(confs, ys, sty, markersize=3, linewidth=1.2, label=SHORT[seq])
    ax.set_xlabel("detector confidence threshold")
    ax.set_ylabel("HOTA")
    ax.legend(ncol=2, fontsize=6.5)
    save(fig, "fig_hota_vs_conf.pdf")


def fig_money_plot(g, prec):
    xs, ys, names = [], [], []
    for seq in SEQS:
        lo, hi = g.get((seq, "conf015"), np.nan), g.get((seq, "conf045"), np.nan)
        if np.isnan(lo) or np.isnan(hi):
            continue
        xs.append(prec[seq]); ys.append(hi - lo); names.append(SHORT[seq])
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    cols = [ORANGE if y > 0 else BLUE for y in ys]
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.scatter(xs, ys, c=cols, s=28, zorder=3, edgecolor="#333", linewidth=0.4)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    r = float("nan")
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        a, b = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs), max(xs), 20)
        ax.plot(xx, a * xx + b, color=RED, linewidth=1.0, linestyle="--",
                label="r = %.2f" % r)
        ax.legend()
    ax.set_xlabel("per-clip detector precision (conf 0.25)")
    ax.set_ylabel(r"$\Delta$HOTA (conf 0.45 $-$ conf 0.15)")
    save(fig, "fig_money_plot.pdf")
    return r


def fig_ablation(g):
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    x = np.arange(len(SEQS)); w = 0.38
    modern = [g.get((s, "conf025"), np.nan) for s in SEQS]
    restored = [g.get((s, "restored_gate"), np.nan) for s in SEQS]
    ax.bar(x - w / 2, modern, w, label="modern (no gate)", color=BLUE, edgecolor="#333", linewidth=0.4)
    ax.bar(x + w / 2, restored, w, label="restored gate", color=ORANGE, edgecolor="#333", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in SEQS], rotation=25, ha="right", fontsize=6.5)
    ax.set_ylabel("HOTA"); ax.legend()
    save(fig, "fig_ablation.pdf")
    rec = [g.get((s, "restored_gate"), np.nan) - g.get((s, "conf025"), np.nan) for s in DENSE]
    rec = [v for v in rec if v == v]
    return float(np.mean(rec)) if rec else float("nan")


def fig_imgsz_dense(g):
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    sizes = [960, 1280, 1536]
    for seq in DENSE:
        ys = [g.get((seq, "imgsz960"), np.nan), g.get((seq, "conf025"), np.nan),
              g.get((seq, "imgsz1536"), np.nan)]
        ax.plot(sizes, ys, "-o", markersize=3.5, linewidth=1.3, label=SHORT[seq])
    ax.set_xlabel("detector input size (px)"); ax.set_ylabel("HOTA"); ax.legend()
    save(fig, "fig_imgsz_dense.pdf")


def update_macros(nhurt, r, recover):
    if not os.path.exists(MACROS):
        return
    s = open(MACROS, encoding="utf-8", newline="").read()
    repl = {"nHurt": str(nhurt),
            "corrPearson": ("%.2f" % r) if r == r else "TBD",
            "ablationRecover": ("+%.2f" % recover) if recover == recover else "TBD"}
    for name, val in repl.items():
        s = re.sub(r"(\\newcommand\{\\%s\}\{)[^}]*(\})" % name, r"\g<1>" + val + r"\g<2>", s)
    open(MACROS, "w", encoding="utf-8", newline="").write(s)
    print("updated", MACROS)


def main():
    g = load_gating()
    prec = load_precision()
    if not g:
        print("gating_study.csv not found or empty; run the Colab study first.")
        return
    fig_hota_vs_conf(g)
    r = fig_money_plot(g, prec)
    recover = fig_ablation(g)
    fig_imgsz_dense(g)
    nhurt = sum(1 for s in SEQS
                if (g.get((s, "conf045"), float("nan")) - g.get((s, "conf015"), float("nan"))) > 0)
    print("\n=== macros ===")
    print("nHurt =", nhurt, "/ 6   (videos where raising confidence/precision raises HOTA)")
    print("corrPearson =", "%.3f" % r if r == r else "n/a",
          "  (corr of per-clip precision with delta-HOTA)")
    print("ablationRecover =", "%.3f" % recover if recover == recover else "n/a",
          "  (mean HOTA recovered by restoring the gate on dense clips)")
    update_macros(nhurt, r, recover)


if __name__ == "__main__":
    main()
