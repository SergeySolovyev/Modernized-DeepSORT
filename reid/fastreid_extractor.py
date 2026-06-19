"""FastReID extractor (JDAI-CV/fast-reid) - optional extra REID source.

FastReID is installed from source. This wraps its DefaultPredictor with a config +
weights. Optional: torchreid + timm + mars already give >=3 models from >=2 sources;
enabling FastReID adds a 4th source for extra diversity. Configure via a reid preset:
  reid: {name: fastreid, config: <yaml>, weights: <pth>, input_size: [256,128]}
"""
import numpy as np

from .base import BaseReIDExtractor
from .registry import register_reid


@register_reid("fastreid")
class FastReIDExtractor(BaseReIDExtractor):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        import cv2  # noqa: F401  (imported lazily in _extract_raw too)
        from fastreid.config import get_cfg
        from fastreid.engine import DefaultPredictor

        fr_cfg = get_cfg()
        fr_cfg.merge_from_file(self.cfg["config"])
        if self.cfg.get("weights"):
            fr_cfg.MODEL.WEIGHTS = self.cfg["weights"]
        fr_cfg.MODEL.DEVICE = str(device)
        fr_cfg.freeze()
        self.predictor = DefaultPredictor(fr_cfg)
        # FastReID expects (H, W); pull from cfg for resizing.
        self.h, self.w = self.cfg.get("input_size", list(fr_cfg.INPUT.SIZE_TEST) or [256, 128])

    def _extract_raw(self, patches):
        import cv2
        import torch

        feats = []
        for p in patches:
            # FastReID DefaultPredictor expects a BGR image; it resizes internally,
            # but we resize to the configured test size for consistency.
            img = cv2.resize(p, (self.w, self.h))
            out = self.predictor(img)               # (1, D) tensor
            feats.append(out.cpu().numpy().reshape(-1))
        return np.asarray(feats, dtype=np.float32)
