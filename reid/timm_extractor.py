"""Generic timm-backbone REID extractor (second REID source).

Not REID-trained — uses an ImageNet-pretrained backbone with the classifier removed
(num_classes=0 -> global-pooled embedding), L2-normalized. Provides architectural/
source diversity vs torchreid. Manual preprocessing (resize + ImageNet normalize).
"""
import numpy as np

from .base import BaseReIDExtractor
from .registry import register_reid

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@register_reid("timm")
class TimmExtractor(BaseReIDExtractor):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        import timm
        import torch

        self._torch = torch
        self.h, self.w = self.cfg.get("input_size", [256, 128])
        self.batch_size = int(self.cfg.get("batch_size", 64))
        model_name = self.cfg.get("model_name", "mobilenetv3_large_100")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(device).eval()

    def _preprocess(self, patches):
        import cv2
        batch = np.empty((len(patches), self.h, self.w, 3), dtype=np.float32)
        for i, p in enumerate(patches):
            rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.w, self.h)).astype(np.float32) / 255.0
            batch[i] = (rgb - _MEAN) / _STD
        return batch.transpose(0, 3, 1, 2)          # NHWC -> NCHW

    def _extract_raw(self, patches):
        torch = self._torch
        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                chunk = self._preprocess(patches[i:i + self.batch_size])
                tens = torch.from_numpy(chunk).to(self.device)
                out = self.model(tens)
                feats.append(out.float().cpu().numpy())
        return np.concatenate(feats, axis=0)
