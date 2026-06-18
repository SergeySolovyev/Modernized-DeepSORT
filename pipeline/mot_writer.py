"""Accumulate and write tracking results in MOTChallenge format.

Output line format (matches the original deep_sort_app):
    frame, id, bb_left, bb_top, bb_width, bb_height, 1, -1, -1, -1
which is what TrackEval expects under data/trackers/.../<seq>.txt.
"""
import os


class MotResultWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.rows = []

    def append(self, frame_idx, track_id, tlwh):
        x, y, w, h = float(tlwh[0]), float(tlwh[1]), float(tlwh[2]), float(tlwh[3])
        self.rows.append((int(frame_idx), int(track_id), x, y, w, h))

    def write(self):
        out_dir = os.path.dirname(os.path.abspath(self.output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as fh:
            for frame_idx, track_id, x, y, w, h in self.rows:
                fh.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n"
                         % (frame_idx, track_id, x, y, w, h))
        return self.output_file
