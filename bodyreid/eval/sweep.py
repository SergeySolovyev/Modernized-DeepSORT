"""Grid-search REID model x cluster-management params on GT descriptors.

Shortlists with the fast embedding metric, then ranks the shortlist by the assignment
simulation (Fowlkes-Mallows). Output drops 1:1 into bodyreid.config.ReidConfig.

  python -m bodyreid.eval.sweep --npz descriptors_osnet.npz --reid osnet_x1_0
"""
import argparse
import csv
import itertools
import os

import numpy as np

from bodyreid.config import ReidConfig
from bodyreid.eval.cluster_eval import score_assignment, score_embedding


def main():
    ap = argparse.ArgumentParser(description="Sweep REID cluster-management params")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--reid", default="unknown", help="label for the model in output rows")
    ap.add_argument("--match-threshs", nargs="*", type=float, default=[0.2, 0.25, 0.3, 0.35, 0.4])
    ap.add_argument("--new-threshs", nargs="*", type=float, default=[0.4, 0.45, 0.5])
    ap.add_argument("--search-targets", nargs="*", default=["centroid", "per_descriptor"])
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, y_true = data["X"], data["y_true"]

    rows = []
    for mt, nt, target, k in itertools.product(
            args.match_threshs, args.new_threshs, args.search_targets, args.ks):
        if nt < mt:
            continue
        emb = score_embedding(X, y_true, mt)                      # fast shortlist signal
        cfg = ReidConfig(match_thresh=mt, new_thresh=nt, search_target=target,
                         search_rule="knn_vote" if (k > 1 and target == "per_descriptor") else "k1",
                         k=k)
        asg = score_assignment(X, y_true, cfg)
        rows.append({"reid": args.reid, "match_thresh": mt, "new_thresh": nt,
                     "search_target": target, "k": k,
                     "fmi_embedding": round(emb["fmi"], 4),
                     "fmi_assignment": round(asg["fmi"], 4),
                     "n_pred": asg["n_pred"], "n_true": asg["n_true"]})

    rows.sort(key=lambda r: r["fmi_assignment"], reverse=True)
    out = args.out or "sweep_%s.csv" % args.reid
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    best = rows[0]
    print("best:", best)
    print("wrote", out)


if __name__ == "__main__":
    main()
