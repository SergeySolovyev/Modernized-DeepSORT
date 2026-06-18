"""Per-track identity history + time-window majority-vote resolution.

For each active track we keep a history of (timestamp, identity_id, distance) and,
each frame, choose the identity that best represents the track over the last T
seconds via a distance-weighted majority vote, with hysteresis to avoid flicker.
"""
from collections import deque


class IdentityResolver:
    def __init__(self, cfg):
        self.cfg = cfg
        self.history = {}        # tid -> deque[(ts, iid, dist)]
        self.resolved = {}       # tid -> iid
        self.last_strength = {}  # tid -> winner vote confidence [0,1]

    def record(self, tid, ts, iid, dist):
        self.history.setdefault(tid, deque()).append((ts, iid, dist))

    def clear(self, tid):
        """Drop a track's window + resolution (used on conflict reset)."""
        self.history.pop(tid, None)
        self.resolved.pop(tid, None)
        self.last_strength.pop(tid, None)

    def resolve(self, tid, now, fps=None):
        T = self.cfg.T_seconds
        dq = self.history.get(tid)
        if dq is not None:                       # trim entries older than the window
            while dq and dq[0][0] < now - T:
                dq.popleft()
        entries = [e for e in (dq or []) if e[1] is not None]
        if not entries:
            self.last_strength[tid] = 0.0
            return self.resolved.get(tid)

        scores = {}
        for _, iid, dist in entries:
            w = 1.0
            if self.cfg.vote_weight == "distance":
                w = max(0.0, 1.0 - dist / max(self.cfg.match_thresh, 1e-6))
            scores[iid] = scores.get(iid, 0.0) + w
        total = sum(scores.values())
        winner = max(scores, key=scores.get)
        conf = scores[winner] / total if total > 0 else 0.0
        self.last_strength[tid] = conf

        # hysteresis: keep previous resolution when the new winner is weak
        if conf < self.cfg.vote_min_conf and tid in self.resolved:
            return self.resolved[tid]
        self.resolved[tid] = winner
        return winner
