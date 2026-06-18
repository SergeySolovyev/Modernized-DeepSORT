"""Import every adapter module and verify it registers itself.

A ModuleNotFoundError for a known heavy backend (torch/cv2/ultralytics/...) is
acceptable (those deps live in Colab, not in a bare dev env); ANY other import
error (bad relative import, decorator typo, NameError) is a real bug and fails.
Run: py -3.14 tests/test_imports.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEAVY = {"cv2", "torch", "torchvision", "ultralytics", "mmdet", "mmcv",
         "mmengine", "nanodet", "torchreid", "timm", "tensorflow", "fastreid", "faiss"}

ADAPTERS = [
    "detectors.yolo_detector", "detectors.yolo_seg_detector", "detectors.gt_detector",
    "detectors.mmdet_detector", "detectors.nanodet_detector",
    "reid.torchreid_extractor", "reid.timm_extractor", "reid.mars_extractor",
    "reid.fastreid_extractor", "pipeline.frame_source",
]


def main():
    imported, skipped, errors = [], [], []
    for mod in ADAPTERS:
        try:
            importlib.import_module(mod)
            imported.append(mod)
            print("IMPORTED", mod)
        except ModuleNotFoundError as exc:
            root = (exc.name or "").split(".")[0]
            if root in HEAVY:
                skipped.append((mod, root))
                print("SKIP    ", mod, "(missing heavy dep: %s)" % root)
            else:
                errors.append((mod, repr(exc)))
                print("ERROR   ", mod, repr(exc))
        except Exception as exc:  # noqa: BLE001
            errors.append((mod, repr(exc)))
            print("ERROR   ", mod, repr(exc))

    # Registries should now contain everything imported above.
    from detectors.registry import _REGISTRY as DET
    from reid.registry import _REGISTRY as REID
    print("registered detectors:", sorted(DET))
    print("registered reid:", sorted(REID))

    print("\n%d imported, %d skipped, %d errors" % (len(imported), len(skipped), len(errors)))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
