"""Bayesian belief over where EVE is (a probability grid) and what state she is in.

Pure inference: this module never imports the world or the environment.
"""
import math
from functools import lru_cache

import numpy as np

from .sensors import expected_strength, wrap_angle

STATUSES = ("active", "damaged", "dormant")


class Geometry:
    """Pairwise distances/bearings between all cells; cached per grid size."""

    def __init__(self, W: int, H: int):
        ys, xs = np.mgrid[0:H, 0:W]
        self.W, self.H, self.n = W, H, W * H
        self.X, self.Y = xs.ravel().astype(float), ys.ravel().astype(float)
        dx = self.X[None, :] - self.X[:, None]  # row = from cell, column = to cell
        dy = self.Y[None, :] - self.Y[:, None]
        self.D = np.hypot(dx, dy)
        self.A = np.arctan2(dy, dx)

    def idx(self, cell) -> int:
        return int(cell[1]) * self.W + int(cell[0])


@lru_cache(maxsize=8)
def geometry(W: int, H: int) -> Geometry:
    return Geometry(W, H)


def entropy(p: np.ndarray) -> float:
    q = p[p > 0]
    return float(-(q * np.log(q)).sum())


def signal_likelihood(g: Geometry, spec: dict, pos_idx: int, strength: float, bearing: float, eps: float):
    """P(reading | EVE at each cell), as a robust mixture so one spoofed reading cannot
    zero out the truth: (1 - eps) * model + eps * flat."""
    E = expected_strength(spec["S0"], spec["d0"], g.D[pos_idx])
    floor = spec["floor"]
    z = (math.log(max(strength, floor)) - np.log(np.maximum(E, floor))) / spec["sigma_log"]
    dth = wrap_angle(bearing - g.A[pos_idx]) / math.radians(spec["sigma_bearing_deg"])
    dth = np.where(g.D[pos_idx] == 0, 0.0, dth)  # standing on the cell: bearing says nothing
    return (1 - eps) * np.exp(-0.5 * z * z - 0.5 * dth * dth) + eps


class Belief:
    def __init__(self, W: int, H: int, spec: dict, dropout: dict, eps: float = 0.15, clue_reliability: float = 0.7):
        self.W, self.H, self.spec, self.dropout = W, H, spec, dropout
        self.eps, self.clue_rel = eps, clue_reliability
        self.g = geometry(W, H)
        self.p = np.full(self.g.n, 1.0 / self.g.n)
        self.status_p = {"active": 0.5, "damaged": 0.3, "dormant": 0.2}
        self.surprises: list[float] = []

    # ----- updates -----
    def _renormalise(self):
        s = self.p.sum()
        if not np.isfinite(s) or s <= 1e-300:  # contradictory evidence: fall back, don't crash
            self.p[:] = 1.0 / self.g.n
        else:
            self.p /= s

    def apply_signal(self, pos, strength: float, bearing_rad: float) -> float:
        """Bayes update from a reading taken at `pos`. Returns how unexpected it was
        (marginal likelihood; small = suspicious, possibly spoofed)."""
        L = signal_likelihood(self.g, self.spec, self.g.idx(pos), strength, bearing_rad, self.eps)
        marginal = float((self.p * L).sum())
        self.p = self.p * L
        self._renormalise()
        self.surprises.append(marginal)
        return marginal

    def apply_dropout(self, dropped: bool):
        """A missing signal is evidence about EVE's condition, not her position."""
        for s in STATUSES:
            q = self.dropout[s]
            self.status_p[s] *= q if dropped else (1 - q)
        tot = sum(self.status_p.values())
        for s in STATUSES:
            self.status_p[s] = self.status_p[s] / tot if tot > 0 else 1 / 3

    def apply_clue(self, clue: dict):
        """A trail clue points at `target` with spread sigma. It may be a decoy, so it is
        used as a mixture with assumed reliability, not trusted outright."""
        tx, ty = clue["target"]
        d2 = (self.g.X - tx) ** 2 + (self.g.Y - ty) ** 2
        L = (1 - self.clue_rel) + self.clue_rel * np.exp(-0.5 * d2 / max(clue["sigma"], 0.5) ** 2)
        self.p = self.p * L
        self._renormalise()

    def apply_negative_inspect(self, pos):
        """Inspected here and found nothing: rule out EVE within 1 cell."""
        near = (np.abs(self.g.X - pos[0]) <= 1) & (np.abs(self.g.Y - pos[1]) <= 1)
        self.p = np.where(near, 0.0, self.p)
        self._renormalise()

    def mark_wall(self, cell):
        i = self.g.idx(cell)
        if self.p[i] > 0:
            self.p[i] = 0.0
            self._renormalise()

    # ----- summaries -----
    def entropy(self) -> float:
        return entropy(self.p)

    def map_cell(self) -> tuple[int, int]:
        i = int(np.argmax(self.p))
        return (i % self.W, i // self.W)

    def mass_near(self, cell, radius: int = 1) -> float:
        near = (np.abs(self.g.X - cell[0]) <= radius) & (np.abs(self.g.Y - cell[1]) <= radius)
        return float(self.p[near].sum())

    def mass3_all(self) -> np.ndarray:
        """Belief mass in the 3x3 block around every cell (for choosing where to inspect)."""
        grid = self.p.reshape(self.H, self.W)
        padded = np.pad(grid, 1)
        out = np.zeros_like(grid)
        for dy in range(3):
            for dx in range(3):
                out += padded[dy:dy + self.H, dx:dx + self.W]
        return out.ravel()

    def credible_cells(self, mass: float = 0.8) -> list[tuple[int, int]]:
        order = np.argsort(-self.p)
        cum, out = 0.0, []
        for i in order:
            out.append((int(i % self.W), int(i // self.W)))
            cum += self.p[i]
            if cum >= mass:
                break
        return out

    def centroid(self) -> tuple[float, float]:
        return float((self.p * self.g.X).sum()), float((self.p * self.g.Y).sum())

    def status_estimate(self) -> tuple[str, float]:
        s = max(self.status_p, key=self.status_p.get)
        return s, self.status_p[s]

    def snapshot(self) -> np.ndarray:
        return self.p.reshape(self.H, self.W).astype(np.float32)
