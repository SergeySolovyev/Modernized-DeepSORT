"""Download the MOT15/MOT16 evaluation sequences and lay them out for TrackEval.

Produces:
  data/MOT/<seq>/{img1,gt,seqinfo.ini}                         # runner reads here
  data/sequences.json                                          # name -> abs path registry
  data/trackeval/gt/mot_challenge/<BENCH>-train/<seq>/{gt,seqinfo.ini}
  data/trackeval/gt/mot_challenge/seqmaps/<BENCH>-train.txt
  data/trackeval/trackers/mot_challenge/                       # (filled by eval/run_tracking)

Usage:
  python -m data.download_mot                  # all six required sequences
  python -m data.download_mot --keep-zip       # keep the downloaded archives
Colab tip: large zips download faster with `!wget`; this script also works there.
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

from data.mot import (ALL_SEQUENCES, BENCHMARKS, DOWNLOAD_URLS, SEQUENCES,
                      benchmark_of, save_registry, sequences_for)

RAW_DIR = os.path.join("data", "raw")
WORK_DIR = os.path.join("data", "MOT")
TE_GT = os.path.join("data", "trackeval", "gt", "mot_challenge")
TE_TRACKERS = os.path.join("data", "trackeval", "trackers", "mot_challenge")


def _download(url, dest):
    if os.path.exists(dest):
        print("  already downloaded:", dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("  downloading", url)

    def _hook(blocks, bs, total):
        if total > 0 and blocks % 200 == 0:
            print("    %.0f%%" % (100.0 * blocks * bs / total), end="\r")

    urllib.request.urlretrieve(url, dest, _hook)
    print("\n  saved", dest)


def _find_seq_dir(root, name):
    """Locate <name>/ anywhere under root (handles train/ subfolders)."""
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == name and os.path.isdir(os.path.join(dirpath, "img1")):
            return dirpath
    return None


def _copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def setup(sequences, keep_zip=False):
    os.makedirs(WORK_DIR, exist_ok=True)
    registry = {}

    # 1) download + extract each needed benchmark once
    needed_benchmarks = sorted({benchmark_of(s) for s in sequences})
    for bench in needed_benchmarks:
        url = DOWNLOAD_URLS[bench]
        zip_path = os.path.join(RAW_DIR, os.path.basename(url))
        _download(url, zip_path)
        extract_root = os.path.join(RAW_DIR, bench)
        if not os.path.isdir(extract_root):
            print("  extracting", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_root)
        if not keep_zip and os.path.exists(zip_path):
            os.remove(zip_path)

    # 2) place each sequence into data/MOT and the TrackEval GT layout
    for seq in sequences:
        bench = benchmark_of(seq)
        src = _find_seq_dir(os.path.join(RAW_DIR, bench), seq)
        if src is None:
            print("  WARNING: could not find sequence %s under %s" % (seq, bench))
            continue
        work = os.path.join(WORK_DIR, seq)
        _copytree(src, work)
        registry[seq] = os.path.abspath(work)

        # TrackEval GT: <BENCH>-train/<seq>/{gt,seqinfo.ini}
        te_seq = os.path.join(TE_GT, "%s-train" % bench, seq)
        os.makedirs(os.path.join(te_seq, "gt"), exist_ok=True)
        if os.path.exists(os.path.join(src, "gt", "gt.txt")):
            shutil.copy(os.path.join(src, "gt", "gt.txt"), os.path.join(te_seq, "gt", "gt.txt"))
        if os.path.exists(os.path.join(src, "seqinfo.ini")):
            shutil.copy(os.path.join(src, "seqinfo.ini"), os.path.join(te_seq, "seqinfo.ini"))
        print("  ready:", seq)

    # 3) seqmaps per benchmark
    seqmap_dir = os.path.join(TE_GT, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    for bench in needed_benchmarks:
        seqs = [s for s in sequences if benchmark_of(s) == bench]
        with open(os.path.join(seqmap_dir, "%s-train.txt" % bench), "w", encoding="utf-8") as fh:
            fh.write("name\n" + "\n".join(seqs) + "\n")

    os.makedirs(TE_TRACKERS, exist_ok=True)
    save_registry(registry)
    print("\nregistry -> data/sequences.json (%d sequences)" % len(registry))


def parse_args():
    ap = argparse.ArgumentParser(description="Download MOT15/MOT16 eval sequences")
    ap.add_argument("--sequences", nargs="*", default=ALL_SEQUENCES,
                    help="subset of sequence names (default: all six)")
    ap.add_argument("--keep-zip", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup(args.sequences, keep_zip=args.keep_zip)
