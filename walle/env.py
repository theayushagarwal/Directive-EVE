"""The simulator. step(action) returns an observation and NOTHING else about the world.

The agent is built from `briefing` (what the operator tells the robot) and from
observations. EVE's position, her status and the clue locations never leave this file
except through sensor readings.
"""
import math

import numpy as np

from . import actions as A
from .grid import DIRS
from .sensors import sample_signal
from .world import World


class Environment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._world = World(cfg)
        self._rng = np.random.default_rng(int(cfg["seed"]) + 10_007)
        self.reset()

    # ----- public -----
    def briefing(self) -> dict:
        """What the robot is told before launch. Contains no hidden truth."""
        c, w = self.cfg, self._world
        return {
            "name": c["name"],
            "width": w.W,
            "height": w.H,
            "start": list(w.start),
            "stations": [list(s) for s in w.stations],
            "resources": c["resources"],
            "sensors": c["sensors"],
            "dropout": c["dropout"],
            "initial_signal": {**w.initial_signal, "origin": list(w.start)},
            "n_clues": int(c["hidden"]["n_clues"]) if c["hidden"]["clues"] is None else len(c["hidden"]["clues"]),
            "clue_radius": c["hidden"]["clue_radius"],
            "agent_params": c["agent"],
        }

    def reset(self):
        w, res = self._world, self.cfg["resources"]
        self.pos = w.start
        self.capacity = float(res["battery"])
        self.battery = self.capacity
        self.time_left = float(res["time"])
        self.step_n = 0
        self.done = False
        self.outcome = None
        return self._obs(None, True, "start")

    def step(self, action: dict) -> dict:
        if self.done:
            raise RuntimeError("Mission is over; call reset()")
        A.validate(action)
        self.step_n += 1
        t = action["type"]
        extra: dict = {}

        if t == A.FINISH:
            return self._finish("abandoned", action, "agent_finished")

        bat, tim = A.cost(self.cfg["resources"], action)
        if bat > self.battery + 1e-9:
            self.time_left -= 1  # a refused action still burns a tick, so loops can't stall
            return self._check_end(self._obs(action, False, "insufficient_battery"))

        ok, event = True, t.lower()
        self.battery -= bat
        self.time_left -= tim
        w = self._world

        if t == A.MOVE:
            dx, dy = DIRS[action["dir"]]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if w.is_wall(nx, ny):
                event = "bumped"
            else:
                self.pos = (nx, ny)
                event = "moved"
        elif t == A.SCAN_SIGNAL:
            extra = self._scan_signal()
            event = "no_signal" if extra["reading"] is None else "signal"
        elif t == A.SCAN_AREA:
            extra = self._scan_area()
        elif t == A.INSPECT:
            hit = w.eve_within(self.pos, 1)
            extra = {"found": hit, "status": w.eve_status if hit else None}
            event = "found_eve" if hit else "nothing_here"
        elif t == A.RECHARGE:
            if self.pos not in [tuple(s) for s in w.stations]:
                ok, event = False, "not_at_station"
            else:
                self.battery = min(self.capacity, self.battery + float(self.cfg["resources"]["recharge_amount"]))
                event = "recharged"

        obs = self._obs(action, ok, event, extra)
        if extra.get("found"):
            return self._finish("found", action, event, obs)
        return self._check_end(obs)

    # ----- internals -----
    def _scan_signal(self) -> dict:
        w, spec = self._world, self.cfg["sensors"]["signal"]
        if self._rng.random() < self.cfg["dropout"][w.eve_status]:
            return {"reading": None}
        source = w.eve_pos
        if w.decoys and self._rng.random() < w.spoof_prob:
            source = w.decoys[int(self._rng.integers(len(w.decoys)))]
        return {"reading": sample_signal(self._rng, spec, self.pos, source)}

    def _scan_area(self) -> dict:
        area = self.cfg["sensors"]["area"]
        seen = []
        for c in self._world.clues:
            if math.dist(c["pos"], self.pos) <= area["radius"] and self._rng.random() < area["detect_prob"]:
                seen.append({"id": c["id"], "pos": list(c["pos"]), "kind": c["kind"],
                             "target": list(c["target"]), "sigma": c["sigma"]})
        return {"clues": seen}

    def _neighbors(self) -> dict:
        out = {}
        for d, (dx, dy) in DIRS.items():
            x, y = self.pos[0] + dx, self.pos[1] + dy
            if not (0 <= x < self._world.W and 0 <= y < self._world.H):
                out[d] = "edge"
            else:
                out[d] = "wall" if self._world.walls[y, x] else "free"
        return out

    def _obs(self, action, ok, event, extra=None) -> dict:
        obs = {
            "step": self.step_n,
            "action": action,
            "ok": ok,
            "event": event,
            "pos": list(self.pos),
            "battery": round(self.battery, 3),
            "time_left": round(self.time_left, 3),
            "at_station": self.pos in [tuple(s) for s in self._world.stations],
            "neighbors": self._neighbors(),
            "done": self.done,
            "outcome": self.outcome,
        }
        obs.update(extra or {})
        return obs

    def _finish(self, outcome, action, event, obs=None):
        self.done, self.outcome = True, outcome
        obs = obs or self._obs(action, True, event)
        obs.update(done=True, outcome=outcome)
        return obs

    def _check_end(self, obs: dict) -> dict:
        at_station = obs["at_station"]
        if self.battery <= 1e-9 and not at_station:
            return self._finish("stranded", obs["action"], "battery_empty_away_from_station", obs)
        if self.time_left <= 0:
            return self._finish("timeout", obs["action"], "out_of_time", obs)
        return obs

    # ----- evaluation only (used by reports/benchmarks, never by agents) -----
    def ground_truth(self) -> dict:
        w = self._world
        return {
            "eve_pos": list(w.eve_pos),
            "eve_status": w.eve_status,
            "clues": [{"id": c["id"], "pos": list(c["pos"]), "target": list(c["target"]), "decoy": c["decoy"]}
                      for c in w.clues],
            "decoys": [list(d) for d in w.decoys],
            "walls": w.walls.copy(),
        }
