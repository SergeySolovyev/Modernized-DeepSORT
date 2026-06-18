"""Download/lay out the MOT15+MOT16 evaluation sequences for TrackEval.

Robust to sources: tries a `--prefetched` directory first (e.g. an already-extracted
Kaggle download), then falls back to downloading per-benchmark zips (URLs overridable).
A failed download for one benchmark does not abort the others.

Produces:
  data/MOT/<seq>/{img1,gt,seqinfo.ini}                         # runner reads here
  data/sequences.json                                          # name -> abs path registry
  data/trackeval/gt/mot_challenge/<BENCH>-train/<seq>/{gt,seqinfo.ini}
  data/trackeval/gt/mot_challenge/seqmaps/<BENCH>-train.txt
  data/trackeval/trackers/mot_challenge/                       # (filled by eval/run_tracking)

Usage:
  python -m data.download_mot                                  # all six required sequences
  python -m data.download_mot --prefetched ~/mot16_from_kaggle # use a manual download
  python -m data.download_mot --mot16-url https://.../MOT16.zip
Kaggle fallback for MOT16 (official site is often unavailable):
  pip install kagglehub
  python -c "import kagglehub; print(kagglehub.dataset_download('takshmandar/mot16-dataset'))"
  python -m data.download_mot --prefetched <printed path>
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

from data.mot import (ALL_SEQUENCES, DOWNLOAD_URLS, benchmark_of, save_registry)

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


def _find_seq_dir(roots, name):
    """Locate <name>/ (containing img1/) anywhere under any of `roots`."""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _dirnames, _files in os.walk(root):
            if os.path.basename(dirpath) == name and os.path.isdir(os.path.join(dirpath, "img1")):
                return dirpath
    return None


def _copytree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def setup(sequences, keep_zip=False, urls=None, prefetched=None):
    urls = dict(DOWNLOAD_URLS, **(urls or {}))
    prefetched = list(prefetched or [])
    os.makedirs(WORK_DIR, exist_ok=True)
    registry = {}

    needed_benchmarks = sorted({benchmark_of(s) for s in sequences})
    for bench in needed_benchmarks:
        bench_seqs = [s for s in sequences if benchmark_of(s) == bench]
        # If the prefetched dirs already contain every needed sequence, skip the download.
        if all(_find_seq_dir(prefetched, s) for s in bench_seqs):
            print("  %s: found in --prefetched, skipping download" % bench)
            continue
        url = urls[bench]
        zip_path = os.path.join(RAW_DIR, os.path.basename(url))
        extract_root = os.path.join(RAW_DIR, bench)
        try:
            _download(url, zip_path)
            if not os.path.isdir(extract_root):
                print("  extracting", zip_path)
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extract_root)
            if not keep_zip and os.path.exists(zip_path):
                os.remove(zip_path)
            prefetched.append(extract_root)
        except Exception as exc:   # one source down must not abort the rest
            print("  WARNING: could not fetch %s from %s (%r).\n"
                  "  Provide --prefetched <dir> with an extracted copy "
                  "(e.g. Kaggle takshmandar/mot16-dataset for MOT16)." % (bench, url, exc))

    search_roots = prefetched + [os.path.join(RAW_DIR, b) for b in needed_benchmarks]
    for seq in sequences:
        src = _find_seq_dir(search_roots, seq)
        if src is None:
            print("  WARNING: sequence %s not found in any source" % seq)
            continue
        work = os.path.join(WORK_DIR, seq)
        _copytree(src, work)
        registry[seq] = os.path.abspath(work)

        bench = benchmark_of(seq)
        te_seq = os.path.join(TE_GT, "%s-train" % bench, seq)
        os.makedirs(os.path.join(te_seq, "gt"), exist_ok=True)
        gt_src = os.path.join(src, "gt", "gt.txt")
        if os.path.exists(gt_src):
            shutil.copy(gt_src, os.path.join(te_seq, "gt", "gt.txt"))
        ini_src = os.path.join(src, "seqinfo.ini")
        if os.path.exists(ini_src):
            shutil.copy(ini_src, os.path.join(te_seq, "seqinfo.ini"))
        print("  ready:", seq)

    seqmap_dir = os.path.join(TE_GT, "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)
    for bench in needed_benchmarks:
        seqs = [s for s in sequences if benchmark_of(s) == bench and s in registry]
        with open(os.path.join(seqmap_dir, "%s-train.txt" % bench), "w", encoding="utf-8") as fh:
            fh.write("name\n" + "\n".join(seqs) + "\n")

    os.makedirs(TE_TRACKERS, exist_ok=True)
    save_registry(registry)
    print("\nregistry -> data/sequences.json (%d/%d sequences ready)"
          % (len(registry), len(sequences)))


def parse_args():
    ap = argparse.ArgumentParser(description="Download MOT15/MOT16 eval sequences")
    ap.add_argument("--sequences", nargs="*", default=ALL_SEQUENCES,
                    help="subset of sequence names (default: all six)")
    ap.add_argument("--prefetched", nargs="*", default=None,
                    help="dirs to search for already-downloaded sequences (e.g. a Kaggle extract)")
    ap.add_argument("--mot15-url", default=None, help="override the 2DMOT2015 zip URL")
    ap.add_argument("--mot16-url", default=None, help="override the MOT16 zip URL")
    ap.add_argument("--keep-zip", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    url_overrides = {}
    if args.mot15_url:
        url_overrides["MOT15"] = args.mot15_url
    if args.mot16_url:
        url_overrides["MOT16"] = args.mot16_url
    setup(args.sequences, keep_zip=args.keep_zip, urls=url_overrides, prefetched=args.prefetched)
