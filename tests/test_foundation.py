"""Lightweight regression tests for the modernized-DeepSORT spine.

These cover the pure-Python/numpy core (config merge, contracts, the nn_matching
dimension guard) and intentionally avoid heavy deps (torch/cv2/ultralytics) so they
run anywhere, including CI. Run:  py -3.14 tests/test_foundation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_merge():
    from config import build_config
    cfg = build_config(detector="yolo", reid="osnet_x1_0")
    assert cfg["detector"]["name"] == "yolo"
    assert cfg["detector"]["weights"] == "yolov8m.pt"
    assert cfg["reid"]["name"] == "torchreid"
    assert cfg["reid"]["model_name"] == "osnet_x1_0"
    assert cfg["tracker"]["max_age"] == 30          # default preserved


def test_overrides_and_coercion():
    from config import build_config
    cfg = build_config(detector="yolo", reid="resnet50", overrides=[
        "tracker.max_cosine_distance=0.3", "detector.imgsz=960", "tracker.nn_budget=None"])
    assert abs(cfg["tracker"]["max_cosine_distance"] - 0.3) < 1e-9
    assert cfg["detector"]["imgsz"] == 960
    assert cfg["tracker"]["nn_budget"] is None
    assert cfg["reid"]["model_name"] == "resnet50"   # reid preset still applied


def test_detection_result():
    from detectors.base import DetectionResult, xyxy_to_tlwh
    tlwh = xyxy_to_tlwh([[10, 20, 30, 60]])
    assert tlwh.tolist() == [[10, 20, 20, 40]]
    dr = DetectionResult(tlwh, [0.9], [0])
    assert len(dr) == 1 and dr.confidence.dtype == np.float32
    assert len(DetectionResult.empty()) == 0


def test_l2_normalize():
    from reid.base import l2_normalize
    f = l2_normalize(np.array([[3.0, 4.0]]))
    assert abs(np.linalg.norm(f[0]) - 1.0) < 1e-6


def test_registry_unknown_name():
    from detectors.registry import build_detector
    try:
        build_detector({"detector": {"name": "does_not_exist"}})
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown detector")


def test_nn_dimension_guard():
    from deep_sort import nn_matching
    m = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
    m.partial_fit(np.random.rand(2, 512).astype(np.float32), [1, 1], [1])
    try:
        m.partial_fit(np.random.rand(1, 128).astype(np.float32), [1], [1])
    except ValueError:
        return
    raise AssertionError("expected ValueError when feature dim changes mid-run")


if __name__ == "__main__":
    failures = 0
    for name in sorted(n for n in dict(globals()) if n.startswith("test_")):
        fn = globals()[name]
        try:
            fn()
            print("PASS", name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("FAIL", name, "->", repr(exc))
    print("ALL GOOD" if failures == 0 else "%d FAILURE(S)" % failures)
    sys.exit(1 if failures else 0)
