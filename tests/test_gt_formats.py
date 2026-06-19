"""Regression tests for MOT15 (10-col) vs MOT16 (9-col) ground-truth parsing.

MOT15/2DMOT2015 gt.txt is 10 columns where cols 7-9 are 3D world coords (-1), NOT
class/visibility. Reading col 8 as visibility (and dropping vis<0) silently zeroed out
all four MOT15 sequences. These tests lock the fix in GtDetector + prepare_gt_crops.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _write(lines):
    path = os.path.join(tempfile.mkdtemp(), "gt.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def test_gt_detector_mot15_10col_kept():
    from detectors.gt_detector import GtDetector
    gt = _write(["1,1,10,20,30,60,1,-1,-1,-1", "1,2,100,20,30,60,1,-1,-1,-1"])  # MOT15
    det = GtDetector({"name": "gt", "gt_file": gt, "pedestrian_classes": [1], "person_class_id": 1})
    assert len(det.detect(None, 1)) == 2, "MOT15 GT boxes were dropped"


def test_gt_detector_mot16_9col_filters():
    from detectors.gt_detector import GtDetector
    gt = _write(["1,1,10,20,30,60,1,1,1.0",     # kept
                 "1,2,100,20,30,60,1,1,0.1",    # vis 0.1 < 0.3 -> dropped
                 "1,3,200,20,30,60,1,3,1.0"])   # class 3 (not pedestrian) -> dropped
    det = GtDetector({"name": "gt", "gt_file": gt, "pedestrian_classes": [1],
                      "person_class_id": 1, "min_visibility": 0.3})
    assert len(det.detect(None, 1)) == 1, "MOT16 visibility/class filtering wrong"


def test_prepare_gt_crops_parse_mot15():
    from data.prepare_gt_crops import _parse_gt
    gt = _write(["1,1,10,20,30,60,1,-1,-1,-1", "2,1,12,20,30,60,1,-1,-1,-1"])
    by_frame = _parse_gt(gt)
    assert sum(len(v) for v in by_frame.values()) == 2, "MOT15 crops were dropped"


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
