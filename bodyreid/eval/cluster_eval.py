"""Score a REID model's descriptors against GT identities (sklearn clustering metrics).

Two complementary modes:
  embedding : AgglomerativeClustering(cosine, distance_threshold=match_thresh) - directly
              mirrors the DB's "merge if within threshold" rule; scores the embedding space.
  assignment: replay descriptors through a headless IdentityDatabase (bodyreid.search) -
              the truest proxy for live create-vs-match; exercises thresholds/k/gallery.

  python -m bodyreid.eval.cluster_eval --npz descriptors_osnet.npz --match-thresh 0.3
"""
import argparse

import numpy as np

from bodyreid.config import ReidConfig
from bodyreid.search import simulate_assignment


def score_embedding(X, y_true, match_thresh):
    """Agglomerative clustering at the DB's match distance -> clustering metrics."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import (calinski_harabasz_score, fowlkes_mallows_score,
                                 silhouette_score)
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=match_thresh).fit_predict(X)
    n_pred = len(set(labels.tolist()))
    out = {"fmi": float(fowlkes_mallows_score(y_true, labels)),
           "n_pred": n_pred, "n_true": int(len(set(y_true.tolist())))}
    if 1 < n_pred < len(X):
        out["silhouette"] = float(silhouette_score(X, labels, metric="cosine"))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    return out


def score_assignment(X, y_true, cfg):
    """Headless DB replay -> Fowlkes-Mallows of assigned identities vs GT."""
    from sklearn.metrics import fowlkes_mallows_score
    pred, db = simulate_assignment(X, cfg)
    return {"fmi": float(fowlkes_mallows_score(y_true, pred)),
            "n_pred": len(set(pred.tolist())), "n_true": int(len(set(y_true.tolist())))}


def main():
    ap = argparse.ArgumentParser(description="Standalone REID clustering metrics")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--match-thresh", type=float, default=0.30)
    ap.add_argument("--new-thresh", type=float, default=0.45)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    X, y_true = data["X"], data["y_true"]
    cfg = ReidConfig(match_thresh=args.match_thresh, new_thresh=args.new_thresh)

    emb = score_embedding(X, y_true, args.match_thresh)
    asg = score_assignment(X, y_true, cfg)
    print("embedding  :", emb)
    print("assignment :", asg)


if __name__ == "__main__":
    main()
