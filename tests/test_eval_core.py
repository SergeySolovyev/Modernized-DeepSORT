"""Tests for the offline-testable eval core (IoU matching + journal)."""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_iou_matrix():
    from eval.iou import iou_matrix
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [10, 10, 10, 10], [5, 0, 10, 10]]
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1.0) < 1e-9          # identical
    assert m[0, 1] == 0.0                       # disjoint
    assert abs(m[0, 2] - (50.0 / 150.0)) < 1e-9  # half-overlap -> 50/150


def test_match_frame_and_prf1():
    from eval.iou import match_frame, prf1
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    det = [[1, 1, 10, 10], [200, 200, 10, 10], [201, 201, 10, 10]]
    tp, fp, fn = match_frame(gt, det, iou_thr=0.5)
    assert (tp, fp, fn) == (1, 2, 1), (tp, fp, fn)
    p, r, f = prf1(tp, fp, fn)
    assert abs(p - 1 / 3) < 1e-9 and abs(r - 0.5) < 1e-9


def test_journal_schema_expansion():
    from eval.journal import append_row
    import csv
    tmp = os.path.join(tempfile.mkdtemp(), "exp.csv")
    append_row({"component": "detector", "seq": "TUD-Campus", "f1": 0.9}, tmp)
    append_row({"component": "hota", "seq": "MOT16-09", "hota": 55.1}, tmp)  # new keys
    with open(tmp, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert set(["component", "seq", "f1", "hota"]).issubset(rows[0].keys())
    assert rows[0]["f1"] == "0.9" and rows[0]["hota"] == ""   # missing key blank
    assert rows[1]["hota"] == "55.1"


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
