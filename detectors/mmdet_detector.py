"""MMDetection detector via the v3.x DetInferencer API.

Install (Colab): `pip install -U openmim && mim install mmengine "mmcv>=2.0" mmdet`.
`model` is an mmdet model alias (e.g. 'rtmdet_tiny_8xb32-300e_coco') resolvable by
DetInferencer, or a path to a config file; weights null -> mim fetches the checkpoint.
"""
import numpy as np

from .base import BaseDetector, DetectionResult, xyxy_to_tlwh
from .registry import register_detector


@register_detector("mmdet")
class MMDetDetector(BaseDetector):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        from mmdet.apis import DetInferencer

        model = self.cfg.get("model", "rtmdet_tiny_8xb32-300e_coco")
        weights = self.cfg.get("weights")  # None -> auto-download via mim
        self.inferencer = DetInferencer(model=model, weights=weights, device=str(device))

    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        out = self.inferencer(frame_bgr, return_datasamples=False,
                              no_save_pred=True, no_save_vis=True, print_result=False)
        preds = out["predictions"][0]
        bboxes = np.asarray(preds.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(preds.get("scores", []), dtype=np.float32).reshape(-1)
        labels = np.asarray(preds.get("labels", []), dtype=np.int64).reshape(-1)
        if bboxes.shape[0] == 0:
            return DetectionResult.empty()
        keep = (labels == self.person_class_id) & (scores >= self.conf)
        if not np.any(keep):
            return DetectionResult.empty()
        return DetectionResult(xyxy_to_tlwh(bboxes[keep]), scores[keep], labels[keep])
