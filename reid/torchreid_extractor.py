"""Torchreid REID extractor (KaiyangZhou/deep-person-reid).

Wraps torchreid's FeatureExtractor. Models: osnet_x1_0 (512-d), osnet_ain_x1_0,
resnet50 (2048-d), mlfn, etc. FeatureExtractor accepts a list of numpy images in
RGB, so BGR crops are converted before the call.
"""
import numpy as np

from .base import BaseReIDExtractor
from .registry import register_reid


def _import_feature_extractor():
    try:                                            # torchreid >= 1.4 layout
        from torchreid.utils import FeatureExtractor
        return FeatureExtractor
    except Exception:                               # some installs nest under .reid
        from torchreid.reid.utils import FeatureExtractor
        return FeatureExtractor


@register_reid("torchreid")
class TorchreidExtractor(BaseReIDExtractor):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        FeatureExtractor = _import_feature_extractor()
        model_name = self.cfg.get("model_name", "osnet_x1_0")
        weights = self.cfg.get("weights") or ""
        h, w = self.cfg.get("input_size", [256, 128])
        self.batch_size = int(self.cfg.get("batch_size", 64))
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=weights,
            image_size=(int(h), int(w)),
            device=str(device),
        )

    def _extract_raw(self, patches):
        rgb = [p[:, :, ::-1] for p in patches]      # BGR -> RGB
        feats = []
        for i in range(0, len(rgb), self.batch_size):
            out = self.extractor(rgb[i:i + self.batch_size])
            feats.append(out.cpu().numpy())
        return np.concatenate(feats, axis=0)
