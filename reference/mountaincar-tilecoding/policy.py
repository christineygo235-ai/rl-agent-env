"""Greedy policy over tile-coded linear Q (DIR/weights.npy + DIR/tiles.json)."""

import json

import numpy as np
from tiles import TileCoder


class GreedyTiles:
    def __init__(self, coder, w):
        self.coder = coder
        self.w = w

    def act(self, obs):
        return int(self.coder.q_values(self.w, obs).argmax())


def load(path):
    cfg = json.loads(open(f"{path}/tiles.json").read())
    coder = TileCoder(int(cfg["n_tilings"]), int(cfg["n_tiles"]))
    w = np.load(f"{path}/weights.npy", allow_pickle=False)
    if w.shape[:3] != coder.shape(1)[:3]:
        raise ValueError(f"weights shape {w.shape} does not match the tile coder")
    return GreedyTiles(coder, w)
