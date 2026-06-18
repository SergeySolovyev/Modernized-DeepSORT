"""boxmot REID extractor — real OSNet (and other) weights with a clean pip install.

Used because KaiyangZhou's torchreid does not build on current Colab (numpy 2.x /
setuptools). boxmot (`pip install boxmot`) bundles OSNet/LMBN/CLIP-ReID weights with
auto-download and a simple feature API: `ReID(...)(crops) -> (N, D) L2-normalized`.

Model is selected by the weights name, e.g. osnet_x1_0_msmt17.pt, osnet_x0_25_msmt17.pt,
osnet_ain_x1_0_msmt17.pt — so multiple REID models come from this one clean source.
"""
import numpy as np

from .base import BaseReIDExtractor
from .registry import register_reid


@register_reid("boxmot")
class BoxmotReID(BaseReIDExtractor):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        from boxmot.reid import ReID

        weights = self.cfg.get("weights", "osnet_x1_0_msmt17.pt")
        dev = "cuda:0" if str(device) == "cuda" else str(device)
        half = bool(self.cfg.get("half", False)) and "cpu" not in dev
        self._reid = ReID(weights=weights, device=dev, half=half)

    def _extract_raw(self, patches):
        # boxmot __call__ accepts a batch of crop arrays (BGR) and returns
        # L2-normalized features; base.extract re-normalizes (harmless).
        feats = self._reid(list(patches))
        return np.asarray(feats, dtype=np.float32)
