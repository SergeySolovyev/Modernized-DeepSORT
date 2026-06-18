"""NanoDet / NanoDet-Plus detector (RangiLyu/nanodet).

NanoDet has no pip API; it is installed from source (pinned commit) and driven by
a config object. This adapter mirrors the official demo `Predictor` inference path.
Install: clone https://github.com/RangiLyu/nanodet into third_party/nanodet,
`pip install -r requirements.txt && python setup.py develop`, and place the config +
weights at the paths in configs/detectors/nanodet.yaml. See data/README.md.

NanoDet returns results as {img_id: {class_id: [[x1,y1,x2,y2,score], ...]}}.
"""
import numpy as np
import torch

from .base import BaseDetector, DetectionResult, xyxy_to_tlwh
from .registry import register_detector


@register_detector("nanodet")
class NanoDetDetector(BaseDetector):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        from nanodet.data.batch_process import stack_batch_img
        from nanodet.data.collate import naive_collate
        from nanodet.data.transform import Pipeline
        from nanodet.model.arch import build_model
        from nanodet.util import Logger, cfg as nanodet_cfg, load_config, load_model_weight

        self._stack_batch_img = stack_batch_img
        self._naive_collate = naive_collate

        load_config(nanodet_cfg, self.cfg["config"])
        self.nanodet_cfg = nanodet_cfg
        logger = Logger(-1, use_tensorboard=False)

        model = build_model(nanodet_cfg.model)
        ckpt = torch.load(self.cfg["weights"], map_location=lambda s, loc: s)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(nanodet_cfg.data.val.pipeline,
                                 nanodet_cfg.data.val.keep_ratio)
        self.input_size = nanodet_cfg.data.val.input_size

    def detect(self, frame_bgr, frame_idx=None) -> DetectionResult:
        img = frame_bgr
        img_info = {"id": 0, "height": img.shape[0], "width": img.shape[1]}
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.input_size)
        meta["img"] = (torch.from_numpy(meta["img"].transpose(2, 0, 1))
                       .to(self.device).float())
        meta = self._naive_collate([meta])
        meta["img"] = self._stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)

        # results: {img_id: {class_id: [[x1,y1,x2,y2,score], ...]}}
        per_class = list(results.values())[0]
        rows = per_class.get(self.person_class_id, [])
        rows = [r for r in rows if len(r) >= 5 and r[4] >= self.conf]
        if not rows:
            return DetectionResult.empty()
        arr = np.asarray(rows, dtype=np.float32)
        xyxy, scores = arr[:, :4], arr[:, 4]
        cls = np.full((len(arr),), self.person_class_id, dtype=np.int64)
        return DetectionResult(xyxy_to_tlwh(xyxy), scores, cls)
