"""Extract person crops from a frame given tlwh boxes.

Returns exactly one crop per box (alignment with the detection list must be
preserved). Boxes are clipped to image bounds; degenerate boxes yield a 1px
patch so the REID extractor - which resizes internally - never receives an
empty array.
"""
import numpy as np


def crop_patches(frame, tlwh_boxes, pad=0.0):
    """Crop `frame` (BGR HxWx3) at each [x, y, w, h] box.

    Parameters
    ----------
    pad : float
        Optional fractional padding added on each side (e.g. 0.1 adds 10% of
        the box size) - sometimes helps REID by including context.

    Returns
    -------
    List[np.ndarray]  one BGR crop per input box.
    """
    h, w = frame.shape[:2]
    patches = []
    for box in np.asarray(tlwh_boxes, dtype=np.float32).reshape(-1, 4):
        x, y, bw, bh = box
        if pad:
            x -= bw * pad
            y -= bh * pad
            bw *= (1.0 + 2.0 * pad)
            bh *= (1.0 + 2.0 * pad)
        x1 = int(max(0, np.floor(x)))
        y1 = int(max(0, np.floor(y)))
        x2 = int(min(w, np.ceil(x + bw)))
        y2 = int(min(h, np.ceil(y + bh)))
        if x2 <= x1:
            x2 = min(w, x1 + 1)
        if y2 <= y1:
            y2 = min(h, y1 + 1)
        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            patch = np.zeros((1, 1, 3), dtype=frame.dtype)
        patches.append(patch)
    return patches
