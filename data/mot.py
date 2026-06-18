"""MOTChallenge sequence constants + path resolution.

The assignment's six evaluation sequences (all in TRAIN splits, so GT is available):
  MOT15 (2DMOT2015): TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1
  MOT16:             MOT16-09, MOT16-11
"""
import json
import os

# name -> TrackEval benchmark (split is always 'train' here).
SEQUENCES = {
    "TUD-Campus": "MOT15",
    "TUD-Stadtmitte": "MOT15",
    "KITTI-17": "MOT15",
    "PETS09-S2L1": "MOT15",
    "MOT16-09": "MOT16",
    "MOT16-11": "MOT16",
}

ALL_SEQUENCES = list(SEQUENCES.keys())
BENCHMARKS = sorted(set(SEQUENCES.values()))

# Default download URLs (overridable via download_mot.py --mot15-url/--mot16-url).
# motchallenge.net is frequently UNREACHABLE (confirmed from Colab: "Network is
# unreachable"), so we default to the PaddleDetection (Baidu) mirror, which is reachable
# and preserves the MOTChallenge layout (images/train/<seq>/{img1,gt,seqinfo.ini}).
DOWNLOAD_URLS = {
    "MOT15": "https://bj.bcebos.com/v1/paddledet/data/mot/MOT15.zip",
    "MOT16": "https://bj.bcebos.com/v1/paddledet/data/mot/MOT16.zip",
}
# Official source (often down): MOT15 https://motchallenge.net/data/2DMOT2015.zip,
#                               MOT16 https://motchallenge.net/data/MOT16.zip

_REGISTRY_FILE = os.path.join("data", "sequences.json")


def benchmark_of(name):
    return SEQUENCES.get(name)


def sequences_for(benchmark):
    return [s for s, b in SEQUENCES.items() if b == benchmark]


def load_registry(registry_file=_REGISTRY_FILE):
    """Return {seq_name: abs_path} written by download_mot.py, or {} if absent."""
    if os.path.exists(registry_file):
        with open(registry_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_registry(mapping, registry_file=_REGISTRY_FILE):
    os.makedirs(os.path.dirname(registry_file) or ".", exist_ok=True)
    with open(registry_file, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2)


def resolve_sequence_dir(name, data_root="data/MOT", registry_file=_REGISTRY_FILE):
    """Locate a sequence directory by name.

    Order: explicit registry (data/sequences.json) -> <data_root>/<name>.
    """
    reg = load_registry(registry_file)
    if name in reg and os.path.isdir(reg[name]):
        return reg[name]
    candidate = os.path.join(data_root, name)
    return candidate
