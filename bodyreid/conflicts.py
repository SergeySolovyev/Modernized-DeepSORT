"""Cross-track identity conflict detection + resolution.

One identity (iid) claimed by two or more simultaneously-active tracks is physically
impossible (one body per identity per instant). Policies:
  reset_all       -> null the iid of every conflicting track (conservative).
  keep_best       -> the highest-confidence track keeps the iid; losers reset & re-search.
  keep_best_spawn -> like keep_best but losers are eligible to spawn a fresh identity
                     on their next miss (here: reset + cleared so re-acquisition fires).
"""


class ConflictManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def resolve(self, resolved, strengths, resolver):
        """resolved: {tid: iid|None}; strengths: {tid: conf}. Mutates + returns resolved."""
        by_iid = {}
        for tid, iid in resolved.items():
            if iid is None:
                continue
            by_iid.setdefault(iid, []).append(tid)

        policy = self.cfg.conflict_policy
        for iid, tids in by_iid.items():
            if len(tids) < 2:
                continue
            if policy == "reset_all":
                for tid in tids:
                    resolved[tid] = None
                    resolver.clear(tid)
                continue
            winner = max(tids, key=lambda t: strengths.get(t, 0.0))
            for tid in tids:
                if tid == winner:
                    continue
                resolved[tid] = None
                resolver.clear(tid)   # forget window so it re-acquires (or spawns) cleanly
        return resolved
