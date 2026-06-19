"""Run the remaining study pieces for a complete report (intended for Colab GPU).

Best-effort: a piece that fails (e.g. an uninstalled detector backend) prints its error and
the rest continue. Covers:
  1. detector P/R/F1 for nanodet and mmdet (the two not yet measured);
  2. standalone clustering for osnet, timm and mars (sklearn FMI / Silhouette / Calinski-Harabasz);
  3. body-REID parameter sweep on a subsample of the osnet descriptors (the full 21k-crop set is
     too slow for a 60-config agglomerative grid).

  python -m eval.finalize_study            # cuda
  python -m eval.finalize_study cpu
"""
import os
import subprocess
import sys

import numpy as np

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"


def run(cmd):
    print("\n$ python -m " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-m"] + cmd, check=False)


def main():
    print("\n########## 1) DETECTORS: nanodet, mmdet (P/R/F1 vs GT) ##########")
    for det in ["nanodet", "mmdet"]:
        print("\n----- det_eval %s -----" % det)
        run(["eval.det_eval", "--detector", det, "--device", DEVICE])

    print("\n########## 2) GT crops (idempotent) ##########")
    run(["data.prepare_gt_crops"])

    print("\n########## 3) STANDALONE CLUSTERING: osnet, timm, mars ##########")
    for reid in ["osnet", "timm_mobilenet", "mars"]:
        npz = "descriptors_%s.npz" % reid
        print("\n----- %s: extract + cluster -----" % reid)
        run(["bodyreid.eval.extract_gt", "--reid", reid, "--out", npz, "--device", DEVICE])
        if os.path.exists(npz):
            run(["bodyreid.eval.cluster_eval", "--npz", npz])
        else:
            print("  (skipped cluster_eval: %s was not produced)" % npz)

    print("\n########## 4) BODY-REID PARAMETER SWEEP (osnet, subsampled) ##########")
    src = "descriptors_osnet.npz"
    if os.path.exists(src):
        d = np.load(src, allow_pickle=True)
        X, y = d["X"], d["y_true"]
        if len(X) > 5000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X), 5000, replace=False)
            X, y = X[idx], y[idx]
            print("subsampled %d -> 5000 crops for the sweep" % len(d["X"]))
        np.savez("descriptors_osnet_sub.npz", X=X, y_true=y)
        run(["bodyreid.eval.sweep", "--npz", "descriptors_osnet_sub.npz", "--reid", "osnet"])
    else:
        print("  (skipped sweep: descriptors_osnet.npz not found)")

    print("\n########## DONE ##########")


if __name__ == "__main__":
    main()
