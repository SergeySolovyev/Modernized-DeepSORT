"""Offline tests for the standalone body-REID identity system (numpy only)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bodyreid.config import ReidConfig            # noqa: E402
from bodyreid.identity_db import IdentityDatabase  # noqa: E402
from bodyreid.search import simulate_assignment    # noqa: E402


def _n(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


A1, A2, B1 = _n([1, 0, 0, 0]), _n([0.97, 0.06, 0, 0]), _n([0, 1, 0, 0])


def test_db_create_match_miss():
    db = IdentityDatabase(ReidConfig())
    iid = db.create(A1, ts=0.0, tid=1)
    hit_iid, d, status = db.search(A2)            # close to A1 -> hit same identity
    assert status == "hit" and hit_iid == iid, (status, hit_iid, d)
    miss_iid, d2, status2 = db.search(B1)         # orthogonal -> miss (signal new)
    assert status2 == "miss" and miss_iid is None, (status2, d2)


def test_centroid_running_mean():
    db = IdentityDatabase(ReidConfig())
    iid = db.create(A1, 0.0, 1)
    db.maybe_enroll(iid, A2, 1.0, tid=1, quality=1.0, status="hit")
    c = db.identities[iid].centroid()
    assert abs(np.linalg.norm(c) - 1.0) < 1e-5     # centroid stays unit-norm
    assert c[0] > 0.9                               # dominated by the A direction


def test_simulate_assignment_two_clusters():
    rng = np.random.RandomState(0)
    X = []
    for _ in range(10):
        X.append(_n(A1 + 0.02 * rng.randn(4)))
    for _ in range(10):
        X.append(_n(B1 + 0.02 * rng.randn(4)))
    pred, db = simulate_assignment(np.stack(X), ReidConfig())
    assert len(set(pred.tolist())) == 2, set(pred.tolist())
    assert len(set(pred[:10])) == 1 and len(set(pred[10:])) == 1   # clean split


# --- minimal fakes to exercise BodyReidRunner without the real tracker ---
class _Det:
    def __init__(self, tlwh, conf=0.9):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = conf


class _Track:
    def __init__(self, tid, tlwh):
        self.track_id = tid
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.time_since_update = 0

    def is_confirmed(self):
        return True

    def to_tlwh(self):
        return self._tlwh.copy()


class _Tracker:
    def __init__(self, tracks):
        self.tracks = tracks


def test_runner_distinct_then_heals_fragmentation():
    from bodyreid.pipeline import BodyReidRunner
    box_a = [10, 10, 60, 130]
    box_b = [400, 10, 60, 130]
    br = BodyReidRunner(ReidConfig())

    # frames 1..5: two distinct people, tids 1 and 2
    for f in range(1, 6):
        dets = [_Det(box_a), _Det(box_b)]
        feats = np.stack([A1, B1])
        trk = _Tracker([_Track(1, box_a), _Track(2, box_b)])
        resolved = br.process_frame(f, None, dets, trk, feats, fps=30)
    iid_a, iid_b = resolved[1], resolved[2]
    assert iid_a is not None and iid_b is not None and iid_a != iid_b, resolved

    # person A re-enters as a NEW track id 3 (tracker fragmentation) with A appearance
    for f in range(6, 11):
        dets = [_Det(box_a)]
        feats = np.stack([A2])                      # same person, slightly different view
        trk = _Tracker([_Track(3, box_a)])
        resolved = br.process_frame(f, None, dets, trk, feats, fps=30)
    assert resolved[3] == iid_a, ("fragmentation not healed", resolved, iid_a)


def test_conflict_keep_best():
    from bodyreid.conflicts import ConflictManager
    from bodyreid.resolver import IdentityResolver
    cfg = ReidConfig()
    res = IdentityResolver(cfg)
    res.resolved = {1: 7, 2: 7}
    res.last_strength = {1: 0.9, 2: 0.6}
    out = ConflictManager(cfg).resolve({1: 7, 2: 7}, res.last_strength, res)
    assert out[1] == 7 and out[2] is None, out      # stronger track keeps it


if __name__ == "__main__":
    failures = 0
    for name in sorted(n for n in dict(globals()) if n.startswith("test_")):
        try:
            globals()[name]()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL", name, "->", repr(exc))
    print("ALL GOOD" if not failures else "%d FAILURE(S)" % failures)
    sys.exit(1 if failures else 0)
