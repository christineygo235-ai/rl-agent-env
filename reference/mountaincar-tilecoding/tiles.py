"""Grid tile coding for MountainCar (Sutton & Barto, section 9.5.4 / chapter 10).

n_tilings grids of n_tiles x n_tiles over (position, velocity), each offset by
an asymmetric displacement (1, 3) / n_tilings of a tile width, as recommended in
S&B. Q(s, a) = sum over tilings of w[tiling, i, j, a].
"""

import numpy as np

LOW = np.array([-1.2, -0.07])
HIGH = np.array([0.6, 0.07])


class TileCoder:
    def __init__(self, n_tilings: int = 8, n_tiles: int = 8):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.scale = n_tiles / (HIGH - LOW)
        displacement = np.array([1.0, 3.0])
        self.offsets = (np.arange(n_tilings)[:, None] * displacement[None, :] / n_tilings) % 1.0
        self.rows = np.arange(n_tilings)

    def shape(self, n_actions: int):
        return (self.n_tilings, self.n_tiles + 1, self.n_tiles + 1, n_actions)

    def indices(self, x):
        coords = (np.asarray(x, dtype=float) - LOW) * self.scale
        cells = np.floor(coords[None, :] + self.offsets).astype(int)
        cells = np.clip(cells, 0, self.n_tiles)
        return self.rows, cells[:, 0], cells[:, 1]

    def q_values(self, w, x):
        r, i, j = self.indices(x)
        return w[r, i, j].sum(axis=0)
