"""Shared eval conventions: sequence->benchmark map and TrackEval data layout.

Layout matches data/download_mot.py:
  GT:       data/trackeval/gt/mot_challenge/<BENCH>-train/<seq>/{gt/gt.txt,seqinfo.ini}
  seqmaps:  data/trackeval/gt/mot_challenge/seqmaps/<BENCH>-train.txt
  trackers: data/trackeval/trackers/mot_challenge/<BENCH>-train/<tracker>/data/<seq>.txt
The TrackEval *code* (run_mot_challenge.py) lives at cfg.paths.trackeval_root.
"""
import os

SEQ_BENCHMARK = {
    "TUD-Campus": "MOT15",
    "TUD-Stadtmitte": "MOT15",
    "KITTI-17": "MOT15",
    "PETS09-S2L1": "MOT15",
    "MOT16-09": "MOT16",
    "MOT16-11": "MOT16",
}
EVAL_SEQUENCES = list(SEQ_BENCHMARK.keys())
SPLIT = "train"

TE_DATA_ROOT = os.path.join("data", "trackeval")
TE_GT = os.path.join(TE_DATA_ROOT, "gt", "mot_challenge")
TE_TRACKERS = os.path.join(TE_DATA_ROOT, "trackers", "mot_challenge")


def benchmark_of(seq):
    return SEQ_BENCHMARK.get(seq)


def tracker_name(detector, reid, mode="live"):
    name = "%s__%s" % (detector, reid)
    return name + ("__gtbox" if mode == "gtbox" else "")


def benchmark_split(bench):
    return "%s-%s" % (bench, SPLIT)


def trackers_data_dir(bench, tracker):
    return os.path.join(TE_TRACKERS, benchmark_split(bench), tracker, "data")


def tracker_result_path(bench, tracker, seq):
    return os.path.join(trackers_data_dir(bench, tracker), seq + ".txt")


def seqmap_path(bench):
    return os.path.join(TE_GT, "seqmaps", "%s.txt" % benchmark_split(bench))
