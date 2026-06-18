"""Parse MOTChallenge `seqinfo.ini` and discover sequence frames.

A MOT sequence directory looks like:
    <seq>/img1/000001.jpg ...
    <seq>/gt/gt.txt
    <seq>/det/det.txt          (optional)
    <seq>/seqinfo.ini          (name, imDir, frameRate, seqLength, imWidth, imHeight, imExt)
"""
import configparser
import glob
import os


def read_seqinfo(sequence_dir):
    """Return a dict of sequence metadata, falling back to scanning img1/.

    Keys: name, im_dir, frame_rate (fps), seq_length, im_width, im_height, im_ext.
    """
    info = {
        "name": os.path.basename(os.path.normpath(sequence_dir)),
        "im_dir": "img1",
        "frame_rate": 30,
        "seq_length": None,
        "im_width": None,
        "im_height": None,
        "im_ext": ".jpg",
    }
    ini_path = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(ini_path):
        parser = configparser.ConfigParser()
        parser.read(ini_path)
        if parser.has_section("Sequence"):
            sec = parser["Sequence"]
            info["name"] = sec.get("name", info["name"])
            info["im_dir"] = sec.get("imDir", info["im_dir"])
            info["frame_rate"] = int(float(sec.get("frameRate", info["frame_rate"])))
            if sec.get("seqLength"):
                info["seq_length"] = int(sec.get("seqLength"))
            if sec.get("imWidth"):
                info["im_width"] = int(sec.get("imWidth"))
            if sec.get("imHeight"):
                info["im_height"] = int(sec.get("imHeight"))
            info["im_ext"] = sec.get("imExt", info["im_ext"])
    return info


def list_frames(sequence_dir, info=None):
    """Return a sorted list of (frame_idx, image_path) for the sequence."""
    info = info or read_seqinfo(sequence_dir)
    image_dir = os.path.join(sequence_dir, info["im_dir"])
    paths = glob.glob(os.path.join(image_dir, "*" + info["im_ext"]))
    if not paths:  # tolerate unexpected extensions
        paths = glob.glob(os.path.join(image_dir, "*.*"))
    frames = []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            frames.append((int(stem), path))
        except ValueError:
            continue
    frames.sort(key=lambda t: t[0])
    return frames
