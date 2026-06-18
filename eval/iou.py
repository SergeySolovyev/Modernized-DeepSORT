"""IoU matching for detector Precision/Recall/F1 against ground truth."""
import numpy as np


def iou_matrix(a_tlwh, b_tlwh):
    """Pairwise IoU between boxes a (N,4) and b (M,4), all in [x,y,w,h]."""
    a = np.asarray(a_tlwh, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_tlwh, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = ((ax2 - ax1) * (ay2 - ay1))[:, None]
    area_b = ((bx2 - bx1) * (by2 - by1))[None, :]
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match_frame(gt_tlwh, det_tlwh, iou_thr=0.5):
    """Greedy one-to-one matching at IoU>=thr. Returns (tp, fp, fn)."""
    gt = np.asarray(gt_tlwh).reshape(-1, 4)
    det = np.asarray(det_tlwh).reshape(-1, 4)
    ng, nd = len(gt), len(det)
    if ng == 0:
        return 0, nd, 0
    if nd == 0:
        return 0, 0, ng
    iou = iou_matrix(det, gt)  # (nd, ng)
    pairs = [(iou[i, j], i, j) for i in range(nd) for j in range(ng) if iou[i, j] >= iou_thr]
    pairs.sort(reverse=True)
    used_d, used_g = set(), set()
    tp = 0
    for _, i, j in pairs:
        if i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        tp += 1
    return tp, nd - len(used_d), ng - len(used_g)


def prf1(tp, fp, fn):
    """Precision, Recall, F1 from accumulated counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1
