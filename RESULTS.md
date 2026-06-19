# Results & Reproducibility

Modernized DeepSORT — person tracking with pluggable detectors and REID.
This note records the **confirmed live result** on Colab Pro and the **exact
copy-pasteable commands** to reproduce it now that the runtime fixes are in.

## 1. Headline result (CONFIRMED, live run)

Benchmark: **MOT16 combined** (MOT16-09 + MOT16-11), HOTA via TrackEval.

| System | HOTA | MOTA | IDF1 | DetA | AssA |
|--------|------|------|------|------|------|
| Unmodified DeepSORT baseline (provided dets + mars-small128) | 38.65 | 46.34 | 50.65 | 37.85 | 39.52 |
| **Modern: YOLOv8m + timm REID** | **47.68** | **46.75** | **52.18** | **50.15** | **45.77** |
| **Delta** | **+9.03** | +0.41 | +1.53 | +12.30 | +6.25 |

**+9.03 HOTA** over the unmodified baseline. The gain is driven by detection
quality (DetA +12.30) and association (AssA +6.25), i.e. better boxes feeding a
stronger appearance metric.

**Real-time:** `fps_bench` on MOT16-09 for `yolo + timm` = **6.35 FPS overall**
(>= 5 FPS requirement met) on a Colab Pro **T4 GPU**.

OSNet (the originally intended best REID) was **build-blocked** under numpy 2.x
(see below) and is **not yet measured**; it would likely score higher than timm.

## 2. Reproduce end-to-end on Colab Pro (copy-paste)

Runtime: **GPU (T4)**, Python 3.12, numpy 2.x — the configuration these fixes
were validated against.

### Option A — the notebook (Run all)

1. Open `notebooks/DeepSORT_Modern.ipynb` in Colab.
2. Runtime -> Change runtime type -> **GPU**.
3. Set `REPO_URL` in cell 1 if your fork differs.
4. **Runtime -> Run all.** It runs install -> download -> baseline -> detector/REID
   studies -> best-combo tracking -> FPS -> body-REID -> overlays -> summary, and
   logs numbers to `results/experiments.csv`.

### Option B — explicit commands

```bash
# --- Clone + enter ---
git clone https://github.com/SergeySolovyev/Modernized-DeepSORT.git
cd Modernized-DeepSORT
git pull -q

# --- Install. Core MUST succeed; the rest are GUARDED so a bleeding-edge ---
# --- runtime build failure cannot break the core pipeline (timm REID always works). ---
pip -q install -r requirements-modern.txt
pip -q install boxmot || echo 'boxmot (OSNet) install failed -> timm REID will be used'
pip -q install -U openmim && mim install mmengine 'mmcv>=2.0' mmdet || echo 'mmdet skipped'
git clone -q https://github.com/JonathonLuiten/TrackEval third_party/TrackEval || true
pip -q install -e third_party/TrackEval

# --- Data (defaults to the reachable PaddleDetection / Baidu MOT mirror) ---
python -m data.download_mot

# --- Baseline: unmodified DeepSORT (provided dets + mars-small128, auto-downloaded) ---
python -m eval.run_baseline --mars third_party/deep_sort_data/mars-small128.pb
python -m eval.trackeval_runner --benchmark MOT16 --trackers baseline

# --- Modern best combo: YOLOv8 + timm REID -> live tracking -> HOTA ---
python -m eval.run_tracking --detector yolo --reid timm --device cuda --per-seq
python -m eval.trackeval_runner --benchmark MOT16 --trackers yolo__timm

# --- Real-time FPS check (>= 5 FPS) ---
python -m eval.fps_bench --detector yolo --reid timm --sequence MOT16-09 --device cuda

# --- Per-video HOTA table (tracker x video + Mean + Delta vs baseline) ---
python -m eval.summarize

# --- Qualitative overlays: baseline vs best (best uses --mot-file to draw stored results) ---
python -m eval.make_overlays --detector gt   --reid mars --sequence MOT16-09 --mode gtbox \
    --out overlays/baseline_MOT16-09.mp4 --device cuda
python -m eval.make_overlays --detector yolo --reid timm --sequence MOT16-09 \
    --mot-file results/MOT16/yolo__timm/data/MOT16-09.txt \
    --out overlays/best_MOT16-09.mp4 --device cuda

# --- Additional task: standalone body-REID identity clustering on GT crops ---
python -m data.prepare_gt_crops
python -m bodyreid.eval.extract_gt   --reid timm --out descriptors_timm.npz --device cuda
python -m bodyreid.eval.cluster_eval --npz descriptors_timm.npz
python -m bodyreid.eval.sweep        --npz descriptors_timm.npz --reid timm
```

> Pass only trackers that actually produced result files to `trackeval_runner`
> (`--trackers ...`). TrackEval **errors** if asked to score a tracker directory
> with no result files, so don't list a combo you didn't run.

## 3. Known-good environment notes

These are the runtime realities discovered by actually running on Colab Pro
(T4, Python 3.12, numpy 2.x); the code now handles each automatically.

- **REID default = `timm`.** `torchreid` (KaiyangZhou/deep-person-reid) and
  `boxmot` both **fail to build** under numpy 2.x (`setup.py egg_info` /
  `metadata-generation-failed`). `timm` installs cleanly and is the **working
  default REID**. boxmot is installed **guarded** (`pip install boxmot || echo ...`)
  so its failure can't break the core install — and it is **not** in
  `requirements-modern.txt`'s `-r` set.
- **OSNet needs a numpy<2 runtime.** Because OSNet ships via torchreid/boxmot,
  measuring it requires pinning a `numpy<2` environment. On stock Colab (numpy 2.x)
  it is build-blocked; the pipeline falls back to timm cleanly.
- **MOT data mirror.** `motchallenge.net` is **unreachable** from Colab. `data/mot.py`
  defaults to the **PaddleDetection / Baidu mirror**
  (`https://bj.bcebos.com/v1/paddledet/data/mot/MOT15.zip` and `MOT16.zip`),
  which preserves the standard MOTChallenge directory layout, so TrackEval and the
  eval scripts work unchanged.
- **mars-small128.pb auto-download.** `eval/run_baseline.py` auto-downloads the
  baseline appearance model `mars-small128.pb` from a GitHub mirror.
- **TrackEval numpy-2.x patch.** TrackEval uses `np.float` (removed in numpy 2.x).
  `eval/trackeval_runner.py` **auto-patches** `np.float` / `np.int` / `np.bool` in
  the TrackEval source before running.
- **TrackEval empty-tracker guard.** TrackEval errors when asked to evaluate a
  tracker with no result files — only pass trackers (`--trackers`) that produced output.

## 4. Pending a clean GPU re-run

The headline numbers above are confirmed live. The following are wired up and
runnable but not yet captured into a committed artifact:

- **Full per-video HOTA table** (`results/experiments.csv` + `report/results_table.md`
  via `python -m eval.summarize`) across all sequences. The `results/` directory is
  not yet populated in the repo — it is produced by a clean GPU run.
- **Overlay videos** (`overlays/baseline_MOT16-09.mp4`, `overlays/best_MOT16-09.mp4`).
- **Body-REID clustering numbers** (`cluster_eval` / `sweep` outputs for the
  standalone identity system).
- **OSNet REID measurement** — requires a `numpy<2` runtime (see above); expected
  to exceed the timm result.
