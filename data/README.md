# Data & backend setup

## MOT sequences (required)
```bash
python -m data.download_mot          # downloads 2DMOT2015 + MOT16, lays out:
#   data/MOT/<seq>/{img1,gt,seqinfo.ini}                         (runner reads here)
#   data/trackeval/gt/mot_challenge/<BENCH>-train/<seq>/...      (TrackEval GT)
#   data/trackeval/gt/mot_challenge/seqmaps/<BENCH>-train.txt
```
The six evaluation sequences (all TRAIN-split, GT available):
MOT15 — TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 ; MOT16 — MOT16-09, MOT16-11.

### MOT16 fallback (official site often unavailable)
`motchallenge.net`'s MOT16 download is frequently unreachable. Use a mirror and point the
script at the extracted folder (`--prefetched` searches it; nested layouts are fine):
```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('takshmandar/mot16-dataset'))"
python -m data.download_mot --prefetched <printed_path>     # MOT15 still downloads normally
```
Or pass a direct mirror URL: `python -m data.download_mot --mot16-url https://.../MOT16.zip`.

## TrackEval (HOTA)
```bash
git clone https://github.com/JonathonLuiten/TrackEval third_party/TrackEval
pip install -e third_party/TrackEval
```
`eval/trackeval_runner.py` invokes `third_party/TrackEval/scripts/run_mot_challenge.py`
against the `data/trackeval/...` layout above.

## GT body crops (for standalone REID eval)
```bash
python -m data.prepare_gt_crops      # -> data/gt_crops/<seq>/<track_id>/<frame>.jpg + manifest.csv
```

## Detector backends
- **YOLO / YOLO-seg** (`ultralytics`): `pip install ultralytics`; weights auto-download.
- **MMDetection** (`mmdet`): `pip install -U openmim && mim install mmengine "mmcv>=2.0" mmdet`.
- **NanoDet** (`nanodet`): clone at a pinned commit and install from source, then place the
  config + checkpoint at the paths in `configs/detectors/nanodet.yaml`:
  ```bash
  git clone https://github.com/RangiLyu/nanodet third_party/nanodet
  cd third_party/nanodet && git checkout <pinned-commit> && pip install -r requirements.txt && python setup.py develop && cd ../..
  # download nanodet-plus-m_416 (config + .pth) into third_party/nanodet/{config,weights}/
  ```

## REID backends
- **torchreid** (`pip install torchreid`): osnet_x1_0 / osnet_ain_x1_0 / resnet50 — weights auto-download.
- **timm** (`pip install timm`): generic backbone, ImageNet-pretrained.
- **mars** (baseline): download the original deep_sort_data and place `mars-small128.pb` at
  `third_party/deep_sort_data/mars-small128.pb` (loaded via `tf.compat.v1`).
- **fastreid** (optional): install from source; point a reid preset at its config + weights.
