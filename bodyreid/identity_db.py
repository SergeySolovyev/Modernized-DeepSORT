"""Persistent identity database for the standalone body-REID system.

Each Identity keeps BOTH a running centroid (from an unnormalized sum, so it is not
corrupted by gallery eviction) AND a capped gallery of recent descriptors (handles
appearance multimodality + enables kNN voting). Matching uses cosine distance
d = 1 - dot on L2-normalized vectors.

Failure-mode mitigations:
  * identity explosion -> creation anchored to track birth (caller), ambiguous-band
    defers premature spawn, offline merge_close(), TTL eviction.
  * centroid drift -> true running mean (raw_sum / n_total), quality-gated admission,
    diversity-aware eviction, ambiguous matches never enrolled.
"""
import numpy as np


class Identity:
    def __init__(self, iid, vec, ts, tid, cfg):
        self.iid = iid
        self.cfg = cfg
        vec = np.asarray(vec, dtype=np.float32)
        self._sum = vec.astype(np.float64).copy()   # unnormalized accumulator
        self.n_total = 1
        self.gallery = []                            # list of (vec, ts, tid, quality)
        self.created_ts = ts
        self.last_ts = ts
        self._stride = {}                            # tid -> admission counter
        self.gallery.append((vec, ts, tid, 1.0))

    def centroid(self):
        c = self._sum / max(self.n_total, 1)
        n = np.linalg.norm(c)
        return (c / n if n > 0 else c).astype(np.float32)

    def add(self, vec, ts, tid, quality):
        vec = np.asarray(vec, dtype=np.float32)
        self._sum += vec
        self.n_total += 1
        self.last_ts = ts
        self.gallery.append((vec, ts, tid, quality))
        if len(self.gallery) > self.cfg.gallery_cap:
            self._evict()

    def _evict(self):
        policy = self.cfg.gallery_evict
        if policy == "fifo":
            self.gallery.pop(0)
        elif policy == "diversity":
            c = self.centroid()
            sims = [float(np.dot(g[0], c)) for g in self.gallery]
            self.gallery.pop(int(np.argmax(sims)))   # drop most redundant
        else:  # "quality": drop lowest-quality
            self.gallery.pop(int(np.argmin([g[3] for g in self.gallery])))


class IdentityDatabase:
    def __init__(self, cfg):
        self.cfg = cfg
        self.identities = {}
        self._next = 1

    # --- lifecycle ---------------------------------------------------------
    def create(self, vec, ts, tid):
        iid = self._next
        self._next += 1
        self.identities[iid] = Identity(iid, vec, ts, tid, self.cfg)
        return iid

    def maybe_enroll(self, iid, vec, ts, tid, quality, status):
        """Enroll a descriptor only on a confident hit, throttled per track."""
        if status != "hit" or iid not in self.identities:
            return
        idn = self.identities[iid]
        count = idn._stride.get(tid, 0)
        idn._stride[tid] = count + 1
        if count % max(self.cfg.gallery_stride, 1) != 0:
            return
        idn.add(vec, ts, tid, quality)

    # --- search ------------------------------------------------------------
    def _bank(self):
        if self.cfg.search_target == "centroid":
            ids = list(self.identities.keys())
            if not ids:
                return np.zeros((0, 1), np.float32), []
            return np.stack([self.identities[i].centroid() for i in ids]), ids
        vecs, ids = [], []
        for iid, idn in self.identities.items():
            for g in idn.gallery:
                vecs.append(g[0])
                ids.append(iid)
        return (np.stack(vecs) if vecs else np.zeros((0, 1), np.float32)), ids

    def _status(self, iid, d):
        if d <= self.cfg.match_thresh:
            return iid, d, "hit"
        if d > self.cfg.new_thresh:
            return None, d, "miss"
        return iid, d, "ambiguous"      # assign tentatively, do not enroll

    def search(self, query):
        """Return (iid_or_None, distance, status in {hit,ambiguous,miss})."""
        if not self.identities:
            return None, 1.0, "miss"
        bank, ids = self._bank()
        if bank.shape[0] == 0:
            return None, 1.0, "miss"
        query = np.asarray(query, dtype=np.float32)
        dists = 1.0 - bank @ query
        order = np.argsort(dists)

        if self.cfg.search_target == "centroid" or self.cfg.search_rule == "k1":
            j = int(order[0])
            return self._status(ids[j], float(dists[j]))

        # per_descriptor voting policies
        if self.cfg.search_rule == "radius":
            sel = [int(k) for k in order if dists[k] <= self.cfg.match_thresh]
        else:  # knn_vote
            sel = [int(k) for k in order[: self.cfg.k] if dists[k] <= self.cfg.new_thresh]
        if not sel:
            return None, float(dists[int(order[0])]), "miss"

        votes = {}
        for k in sel:
            w = max(0.0, 1.0 - dists[k] / max(self.cfg.new_thresh, 1e-6))
            votes[ids[k]] = votes.get(ids[k], 0.0) + w
        iid = max(votes, key=votes.get)
        d = min(float(dists[k]) for k in sel if ids[k] == iid)
        return self._status(iid, d)

    # --- housekeeping ------------------------------------------------------
    def evict_ttl(self, now):
        ttl = self.cfg.identity_ttl_seconds
        dead = [i for i, idn in self.identities.items() if now - idn.last_ts > ttl]
        for i in dead:
            del self.identities[i]
        return dead

    def merge_close(self):
        """Offline: merge identities whose centroids are within merge_thresh and
        whose active spans do not overlap. Returns list of (kept, removed)."""
        merged = []
        ids = list(self.identities.keys())
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ia, ib = ids[a], ids[b]
                if ia not in self.identities or ib not in self.identities:
                    continue
                A, B = self.identities[ia], self.identities[ib]
                d = 1.0 - float(np.dot(A.centroid(), B.centroid()))
                overlap = not (A.last_ts < B.created_ts or B.last_ts < A.created_ts)
                if d <= self.cfg.merge_thresh and not overlap:
                    A._sum += B._sum
                    A.n_total += B.n_total
                    A.gallery.extend(B.gallery)
                    A.last_ts = max(A.last_ts, B.last_ts)
                    del self.identities[ib]
                    merged.append((ia, ib))
        return merged
