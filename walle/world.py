"""Ground truth. ONLY env.py is allowed to import this; the agent never sees it.

Everything not pinned in the config is generated from the seed, so a mission is
fully reproducible from (config, seed).
"""
import math

import numpy as np

from .config import ConfigError
from .grid import bfs_dist, cheb
from .sensors import sample_signal


class World:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        m = cfg["map"]
        self.W, self.H = int(m["width"]), int(m["height"])
        rng = np.random.default_rng(int(cfg["seed"]))
        mv = cfg["resources"]["costs"]["move"]

        self.start = tuple(cfg["start"])
        self.walls = self._make_walls(m, rng)
        self.walls[self.start[1], self.start[0]] = False
        reach = bfs_dist(self.walls, [self.start])
        reachable = np.isfinite(reach)

        self.stations = self._make_stations(cfg, reach, rng)
        for sx, sy in self.stations:
            self.walls[sy, sx] = False
        d_station = bfs_dist(self.walls, self.stations)
        reachable = np.isfinite(d_station)

        hid = cfg["hidden"]
        self.eve_pos, self.eve_status = self._make_eve(hid, cfg, d_station, reachable, rng, mv)
        d_eve = bfs_dist(self.walls, [self.eve_pos])
        self.clues = self._make_clues(hid, reachable, rng)
        self.decoys = self._make_decoys(hid, reachable, rng)
        self.spoof_prob = float(hid["spoof_prob"])
        self.initial_signal = self._make_initial_signal(cfg, rng)
        self.eve_path_len = float(d_station[self.eve_pos[1], self.eve_pos[0]])
        self._d_eve = d_eve

    # ----- generation -----
    def _make_walls(self, m, rng):
        if m.get("walls"):
            walls = np.zeros((self.H, self.W), bool)
            for x, y in m["walls"]:
                if not (0 <= x < self.W and 0 <= y < self.H):
                    raise ConfigError(f"wall {[x, y]} is outside the map")
                walls[y, x] = True
            return walls
        return rng.random((self.H, self.W)) < float(m["wall_density"])

    def _make_stations(self, cfg, reach, rng):
        if cfg["stations"]:
            stations = [tuple(s) for s in cfg["stations"]]
            for s in stations:
                if self.walls[s[1], s[0]]:
                    self.walls[s[1], s[0]] = False
            return stations
        ys, xs = np.nonzero(np.isfinite(reach) & (reach >= 6) & (reach <= 14))
        if len(xs) == 0:
            ys, xs = np.nonzero(np.isfinite(reach) & (reach >= 3))
        if len(xs) == 0:
            return [self.start]
        i = int(rng.integers(len(xs)))
        return [self.start, (int(xs[i]), int(ys[i]))]

    def _free_cells(self, mask):
        ys, xs = np.nonzero(mask)
        return list(zip(xs.tolist(), ys.tolist()))

    def _make_eve(self, hid, cfg, d_station, reachable, rng, mv):
        statuses = ["active", "damaged", "dormant"]
        status_draw = statuses[int(rng.choice(3, p=[0.5, 0.3, 0.2]))]
        eve = hid.get("eve")
        if eve:
            pos = tuple(eve["pos"])
            if not (0 <= pos[0] < self.W and 0 <= pos[1] < self.H) or not reachable[pos[1], pos[0]]:
                raise ConfigError(f"hidden.eve.pos {list(pos)} is a wall or unreachable")
            status = eve.get("status", status_draw)
            if status not in statuses:
                raise ConfigError(f"hidden.eve.status must be one of {statuses}")
            return pos, status
        lo, hi = hid.get("eve_dist_range") or [6, max(8, int(0.35 * cfg["resources"]["battery"] / mv))]
        cand = self._free_cells(reachable & (d_station >= lo) & (d_station <= hi))
        if not cand:
            cand = self._free_cells(reachable & (d_station >= 3))
        if not cand:
            raise ConfigError("No valid place for EVE on this map; lower wall_density or change the seed")
        return cand[int(rng.integers(len(cand)))], status_draw

    def _clip(self, x, y):
        return (int(min(max(round(x), 0), self.W - 1)), int(min(max(round(y), 0), self.H - 1)))

    def _make_clues(self, hid, reachable, rng):
        if hid.get("clues") is not None:
            return [
                {"id": i, "pos": tuple(c["pos"]), "kind": "trail", "target": tuple(c["target"]),
                 "sigma": float(c.get("sigma", hid["clue_sigma"])), "decoy": bool(c.get("decoy", False))}
                for i, c in enumerate(hid["clues"])
            ]
        radius = hid["clue_radius"]
        near = [c for c in self._free_cells(reachable) if 2 <= math.dist(c, self.eve_pos) <= radius]
        far = [c for c in self._free_cells(reachable) if math.dist(c, self.eve_pos) >= 6]
        clues = []
        for i in range(int(hid["n_clues"])):
            if not near:
                break
            pos = near[int(rng.integers(len(near)))]
            decoy = bool(rng.random() < hid["decoy_clue_fraction"])
            if decoy and far:
                target = far[int(rng.integers(len(far)))]
            else:
                decoy = False
                target = self._clip(self.eve_pos[0] + rng.normal(0, hid["clue_sigma"]),
                                    self.eve_pos[1] + rng.normal(0, hid["clue_sigma"]))
            clues.append({"id": i, "pos": pos, "kind": "trail", "target": target,
                          "sigma": float(hid["clue_sigma"]), "decoy": decoy})
        return clues

    def _make_decoys(self, hid, reachable, rng):
        if hid.get("decoys") is not None:
            return [tuple(d) for d in hid["decoys"]]
        cand = [c for c in self._free_cells(reachable) if math.dist(c, self.eve_pos) >= 5]
        out = []
        for _ in range(int(hid["n_decoys"])):
            if cand:
                out.append(cand[int(rng.integers(len(cand)))])
        return out

    def _make_initial_signal(self, cfg, rng):
        if cfg.get("initial_signal"):
            s = cfg["initial_signal"]
            return {"strength": float(s["strength"]), "bearing_deg": float(s["bearing_deg"]),
                    "timestamp": float(s.get("timestamp", 0))}
        r = sample_signal(rng, cfg["sensors"]["signal"], self.start, self.eve_pos)
        r["timestamp"] = 0.0
        return r

    # ----- queries used by the environment -----
    def is_wall(self, x, y) -> bool:
        return not (0 <= x < self.W and 0 <= y < self.H) or bool(self.walls[y, x])

    def eve_within(self, pos, radius=1) -> bool:
        return cheb(pos, self.eve_pos) <= radius
