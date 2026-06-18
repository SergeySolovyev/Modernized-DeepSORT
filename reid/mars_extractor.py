"""Original DeepSORT appearance model (mars-small128) — 128-d BASELINE.

Loads the TF1 frozen graph via tf.compat.v1 (works under TF2 in Colab). The graph
input/output tensors are located by name suffix to tolerate the 'net/' import prefix.
Crops are resized to the model's input shape (no BGR->RGB conversion, matching the
original tools/generate_detections.py encoder). This is the reference REID for the
appearance study; the canonical Step-1 baseline uses the original code path.
"""
import cv2
import numpy as np

from .base import BaseReIDExtractor
from .registry import register_reid


@register_reid("mars")
class MarsExtractor(BaseReIDExtractor):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device)
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

        pb_path = self.cfg["weights"]
        self.batch_size = int(self.cfg.get("batch_size", 32))

        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_path, "rb") as fh:
                graph_def.ParseFromString(fh.read())
            tf.import_graph_def(graph_def, name="net")
            self.input_var = self._find_tensor("images")
            self.output_var = self._find_tensor("features")
        self.session = tf.Session(graph=self.graph)

        # input shape (H, W, C) and output dim from the graph itself
        in_shape = self.input_var.get_shape().as_list()
        self.in_h, self.in_w = int(in_shape[1]), int(in_shape[2])
        self._feature_dim = int(self.output_var.get_shape().as_list()[-1])

    def _find_tensor(self, suffix):
        for op in self.graph.get_operations():
            for out in op.outputs:
                if out.name.endswith("%s:0" % suffix):
                    return out
        raise KeyError("Tensor '*%s:0' not found in mars graph" % suffix)

    def _extract_raw(self, patches):
        data = np.zeros((len(patches), self.in_h, self.in_w, 3), dtype=np.uint8)
        for i, p in enumerate(patches):
            data[i] = cv2.resize(p, (self.in_w, self.in_h))
        out = np.zeros((len(patches), self._feature_dim), dtype=np.float32)
        for s in range(0, len(patches), self.batch_size):
            e = min(s + self.batch_size, len(patches))
            out[s:e] = self.session.run(self.output_var,
                                        feed_dict={self.input_var: data[s:e]})
        return out
