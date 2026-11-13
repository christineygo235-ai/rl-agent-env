"""Greedy tabular policy loader for FrozenLake.

The artefact directory holds q.npy (float array nS x nA) and meta.json which
contains the map description (list of strings) used only for tie-breaking.
"""
import json
import os

import numpy as np


def _as_state(obs):
    if isinstance(obs, (int, np.integer)):
        return int(obs)
    arr = np.asarray(obs).ravel()
    return int(arr[0])


class GreedyPolicy:
    """Greedy action w.r.t. a Q-table.

    Ties in the Q-table are broken by preferring successor states that are not
    holes and that are closer (Manhattan) to the goal.  This only matters in
    states where the Q-table is exactly tied (e.g. rarely visited cells) and
    makes the deterministic greedy policy much more reliable.
    """

    def __init__(self, q, desc=None):
        self.q = np.asarray(q, dtype=np.float64)
        self.nS, self.nA = self.q.shape
        self.holes = None
        self.goals = None
        self.nrow = self.ncol = None
        if desc:
            self.nrow = len(desc)
            self.ncol = len(desc[0])
            self.holes = set()
            self.goals = set()
            for r, row in enumerate(desc):
                for c, ch in enumerate(row):
                    idx = r * self.ncol + c
                    if ch in ("H",):
                        self.holes.add(idx)
                    elif ch in ("G",):
                        self.goals.add(idx)

    def _next_state(self, s, a):
        r, c = divmod(s, self.ncol)
        if a == 0:
            c = max(c - 1, 0)
        elif a == 1:
            r = min(r + 1, self.nrow - 1)
        elif a == 2:
            c = min(c + 1, self.ncol - 1)
        else:
            r = max(r - 1, 0)
        return r * self.ncol + c

    def _tie_break_key(self, s, a):
        # lower is better
        if self.holes is None or not self.goals:
            return (0, 0)
        nxt = self._next_state(s, a)
        gr, gc = divmod(min(self.goals), self.ncol)
        r, c = divmod(nxt, self.ncol)
        return (1 if nxt in self.holes else 0, abs(r - gr) + abs(c - gc))

    def act(self, obs):
        s = _as_state(obs)
        if s < 0 or s >= self.nS:
            return 0
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        if len(best) == 1:
            return int(best[0])
        return int(min(best, key=lambda a: (self._tie_break_key(s, a), int(a))))


def load(path):
    """Load a policy from a directory (or a bare q.npy path)."""
    if os.path.isdir(path):
        qpath = os.path.join(path, "q.npy")
        metapath = os.path.join(path, "meta.json")
    else:
        qpath = path
        metapath = os.path.join(os.path.dirname(path), "meta.json")
    q = np.load(qpath)
    desc = None
    if os.path.exists(metapath):
        try:
            with open(metapath) as f:
                desc = json.load(f).get("desc")
        except Exception:
            desc = None
    return GreedyPolicy(q, desc)
