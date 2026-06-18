"""Iterate frames of a MOTChallenge sequence as (frame_idx, BGR image)."""
import cv2

from data.seqinfo import list_frames, read_seqinfo


class FrameSource:
    """Lazily yields (frame_idx, image_bgr) for a MOT sequence directory.

    Exposes sequence metadata (image_size, frame range, fps) needed by the
    tracker/visualizer and for converting the body-REID time window T to frames.
    """

    def __init__(self, sequence_dir):
        self.sequence_dir = sequence_dir
        self.info = read_seqinfo(sequence_dir)
        self.frames = list_frames(sequence_dir, self.info)
        if not self.frames:
            raise FileNotFoundError("No frames found under %s" % sequence_dir)
        self.name = self.info["name"]
        self.fps = self.info["frame_rate"] or 30
        self.min_frame_idx = self.frames[0][0]
        self.max_frame_idx = self.frames[-1][0]
        # Resolve image size, preferring seqinfo, else reading the first frame.
        if self.info["im_height"] and self.info["im_width"]:
            self.image_size = (self.info["im_height"], self.info["im_width"])
        else:
            first = cv2.imread(self.frames[0][1], cv2.IMREAD_COLOR)
            self.image_size = first.shape[:2] if first is not None else (1080, 1920)

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        for frame_idx, path in self.frames:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            yield frame_idx, image

    def read(self, frame_idx):
        """Read a single frame by index (returns BGR image or None)."""
        for idx, path in self.frames:
            if idx == frame_idx:
                return cv2.imread(path, cv2.IMREAD_COLOR)
        return None
