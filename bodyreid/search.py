"""Search helpers + a headless assignment simulation used by the standalone eval.

The actual nearest-neighbour logic (k1 / knn_vote / radius, centroid vs per-descriptor)
lives in IdentityDatabase.search. This module adds the headless replay used by
bodyreid/eval to tune cluster-management params on GT crops without the tracker.
"""
import numpy as np

from .identity_db import IdentityDatabase


def simulate_assignment(X, cfg, timestamps=None):
    """Replay descriptors X (N,D, L2-normed) through a headless IdentityDatabase.

    Each descriptor is searched; a miss creates a new identity, a hit/ambiguous is
    assigned (hits also enroll). Returns predicted identity labels (N,), the truest
    proxy for the live create-vs-match behaviour (exercises thresholds, k, gallery).
    """
    db = IdentityDatabase(cfg)
    pred = np.empty((len(X),), dtype=np.int64)
    for i, vec in enumerate(X):
        ts = float(timestamps[i]) if timestamps is not None else float(i)
        iid, dist, status = db.search(vec)
        if status == "miss" or iid is None:
            iid = db.create(vec, ts, tid=i)
        else:
            db.maybe_enroll(iid, vec, ts, tid=i, quality=1.0, status=status)
        pred[i] = iid
    return pred, db
