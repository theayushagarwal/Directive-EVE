"""Agents. They see only the briefing and observations, never the world.

BaseAgent owns bookkeeping (position, battery, discovered map, belief) and the safety
rules every agent obeys: top up at a station, and turn back while there is still enough
battery to get home. Subclasses only decide what to do with the spare resources.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import actions as A
from .belief import Belief
from .grid import DIRS, bfs_dist
from .planner import clue_search_value, expected_info_gain, inspect_value

FOUND_VALUE = 6.0     # nats-equivalent worth of actually finding EVE
MIN_INSPECT_MASS = 0.12
CLUE_GAIN = 1.2       # nats a typical trail clue is worth
MAX_EIG_CANDIDATES = 120
REPEAT_DISCOUNT = 0.65  # each earlier scan at the same cell cuts a new scan's value by this factor
RECHARGE_BELOW = 0.85  # top up at a station when below this fraction


@dataclass
class Decision:
    action: dict
    strategy: str
    reason: str
    candidates: list = field(default_factory=list)


class BaseAgent:
    name = "base"

    def __init__(self, briefing: dict, seed: int = 0, params: dict | None = None):
        self.b = briefing
        self.W, self.H = briefing["width"], briefing["height"]
        res = briefing["resources"]
        self.cap = float(res["battery"])
        self.time_budget = float(res["time"])
        self.costs, self.tcosts = res["costs"], res["time_costs"]
        self.stations = [tuple(s) for s in briefing["stations"]]
        self.params = {**briefing["agent_params"], **(params or {})}
        self.rng = np.random.default_rng(seed)

        self.pos = tuple(briefing["start"])
        self.battery, self.time_left = self.cap, self.time_budget
        self.known = np.full((self.H, self.W), -1, dtype=int)  # -1 unknown, 0 free, 1 wall
        self.known[self.pos[1], self.pos[0]] = 0
        self.covered = np.zeros((self.H, self.W), bool)
        self.clues: dict[int, dict] = {}
        self.found = False
        self.found_status = None
        self.stats = dict(moves=0, bumps=0, scans=0, area_scans=0, inspects=0, recharges=0,
                          dropouts=0, suspicious=0)
        self.last_reading = None
        self.scan_counts: dict = {}

        spec = briefing["sensors"]["signal"]
        self.belief = Belief(self.W, self.H, spec, briefing["dropout"],
                             eps=self.params["spoof_epsilon"], clue_reliability=self.params["clue_reliability"])
        s = briefing["initial_signal"]
        self.belief.apply_signal(tuple(s["origin"]), s["strength"], math.radians(s["bearing_deg"]))

    # ----- map helpers -----
    def blocked(self) -> np.ndarray:
        return self.known == 1

    def dist_to_station(self, conservative: bool = False) -> np.ndarray:
        """Steps to the nearest station. Optimistic planning treats unexplored cells as free;
        `conservative` only walks cells already seen to be free, so the route is guaranteed."""
        blocked = (self.known != 0) if conservative else self.blocked()
        return bfs_dist(blocked, self.stations)

    def step_toward(self, targets, conservative: bool = False):
        """First step of a shortest path to the nearest target."""
        blocked = (self.known != 0) if conservative else self.blocked()
        dist = bfs_dist(blocked, targets)
        here = dist[self.pos[1], self.pos[0]]
        if not np.isfinite(here):
            return None
        best, best_d = None, here
        for d, (dx, dy) in DIRS.items():
            x, y = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= x < self.W and 0 <= y < self.H and dist[y, x] < best_d:
                best, best_d = d, dist[y, x]
        return best

    # ----- perception -----
    def observe(self, obs: dict) -> None:
        self.pos = tuple(obs["pos"])
        self.battery, self.time_left = obs["battery"], obs["time_left"]
        self.known[self.pos[1], self.pos[0]] = 0
        for d, state in obs["neighbors"].items():
            dx, dy = DIRS[d]
            x, y = self.pos[0] + dx, self.pos[1] + dy
            if state == "wall":
                self.known[y, x] = 1
                self.belief.mark_wall((x, y))
            elif state == "free":
                self.known[y, x] = 0
        action = obs.get("action")
        if not action or not obs["ok"]:
            return
        t = action["type"]
        if t == A.MOVE:
            self.stats["bumps" if obs["event"] == "bumped" else "moves"] += 1
        elif t == A.SCAN_SIGNAL:
            self.stats["scans"] += 1
            self.scan_counts[self.pos] = self.scan_counts.get(self.pos, 0) + 1
            r = obs["reading"]
            self.last_reading = r
            if r is None:
                self.stats["dropouts"] += 1
                self.belief.apply_dropout(True)
            else:
                marginal = self.belief.apply_signal(self.pos, r["strength"], math.radians(r["bearing_deg"]))
                self.belief.apply_dropout(False)
                if marginal < 0.2:
                    self.stats["suspicious"] += 1
        elif t == A.SCAN_AREA:
            self.stats["area_scans"] += 1
            ys, xs = np.mgrid[0:self.H, 0:self.W]
            rad = self.b["sensors"]["area"]["radius"]
            self.covered |= (xs - self.pos[0]) ** 2 + (ys - self.pos[1]) ** 2 <= rad ** 2
            for c in obs["clues"]:
                if c["id"] not in self.clues:
                    self.clues[c["id"]] = c
                    self.belief.apply_clue(c)
        elif t == A.INSPECT:
            self.stats["inspects"] += 1
            if obs["found"]:
                self.found, self.found_status = True, obs["status"]
            else:
                self.belief.apply_negative_inspect(self.pos)
        elif t == A.RECHARGE:
            self.stats["recharges"] += 1

    # ----- decisions -----
    def decide(self) -> Decision:
        return self.guard(self.safety_override() or self.choose())

    def guard(self, d: Decision) -> Decision:
        """Last line of defence, applied to every agent: refuse any action that would leave
        too little battery to reach a station along cells known to be free."""
        t = d.action["type"]
        if t in (A.RECHARGE, A.FINISH):
            return d
        mv = self.costs["move"]
        cost, _ = A.cost(self.b["resources"], d.action)
        here = self.dist_to_station(conservative=True)[self.pos[1], self.pos[0]]
        if not np.isfinite(here):
            here = self.dist_to_station()[self.pos[1], self.pos[0]]
        worst_after = (here + (1 if t == A.MOVE else 0)) * mv
        if not np.isfinite(here) or self.battery - cost >= worst_after:
            return d
        if self.pos in self.stations:
            if self.time_left > self.tcosts["recharge"]:
                return Decision(A.act(A.RECHARGE), "recharge",
                                "Battery too low to leave the station safely; recharging instead.")
            return Decision(A.act(A.FINISH), "finish",
                            "Battery too low to leave the station safely and no time to recharge; staying put.")
        step = self.step_toward(self.stations, conservative=True) or self.step_toward(self.stations)
        if step:
            return Decision(A.move(step), "return",
                            "That action would leave too little battery to get home; heading to a station instead.")
        return Decision(A.act(A.FINISH), "finish", "No safe action available; ending the mission.")

    def safety_override(self) -> Decision | None:
        mv, margin = self.costs["move"], self.params["safety_margin"]
        at_station = self.pos in self.stations
        if at_station:
            if self.battery < self.cap * RECHARGE_BELOW and self.time_left > self.tcosts["recharge"] + 8:
                return Decision(A.act(A.RECHARGE), "recharge",
                                f"At a charging station with battery {self.battery:.0f}/{self.cap:.0f}: topping up before leaving.")
            return None
        # Turn-back decisions use ONLY cells already seen to be free: the way home is guaranteed
        # and exactly as long as we think, so newly discovered walls cannot eat the margin.
        here = self.dist_to_station(conservative=True)[self.pos[1], self.pos[0]]
        conservative = np.isfinite(here)
        if not conservative:
            here = self.dist_to_station()[self.pos[1], self.pos[0]]
        if np.isfinite(here) and self.battery < here * mv + margin:
            d = self.step_toward(self.stations, conservative=conservative)
            if d:
                return Decision(A.move(d), "return",
                                f"Turning back: battery {self.battery:.0f} barely covers the {here:.0f}-step trip to a "
                                f"station plus a safety margin of {margin}.")
        return None

    def choose(self) -> Decision:
        raise NotImplementedError


class BeliefAgent(BaseAgent):
    """Picks, every step, the action with the best expected information (or chance of
    finding EVE) per unit of battery and time, subject to always being able to get home."""

    name = "belief"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.target = None  # (kind, cell index) we are currently committed to

    def choose(self) -> Decision:
        g, b = self.belief.g, self.belief
        spec = self.b["sensors"]["signal"]
        area = self.b["sensors"]["area"]
        mv, tmv = self.costs["move"], self.tcosts["move"]
        margin = self.params["safety_margin"]
        pos_i = g.idx(self.pos)

        dpos = bfs_dist(self.blocked(), [self.pos]).ravel()
        dst = self.dist_to_station().ravel()
        base_ok = np.isfinite(dpos) & np.isfinite(dst) & (self.known.ravel() != 1)

        def feasible(key):
            bat = dpos * mv + self.costs[key]
            tim = dpos * tmv + self.tcosts[key]
            ok = base_ok & (bat + dst * mv + margin <= self.battery) & (tim <= self.time_left)
            return bat, tim, ok

        scored = {}  # (kind, idx) -> (score, gain, steps)
        norm = lambda bat, tim: bat / self.cap + tim / self.time_budget + 1e-3

        bat, tim, ok = feasible("scan_signal")
        cand = np.nonzero(ok)[0]
        if len(cand) > MAX_EIG_CANDIDATES:
            cand = cand[np.argsort(dpos[cand])[:MAX_EIG_CANDIDATES]]
        if len(cand):
            gains = expected_info_gain(g, b.p, spec, b.eps, cand, self.rng)
            # Readings from one spot share systematic error, and moving to a new vantage point
            # triangulates far better, so repeat scans at the same cell are worth less.
            gains = gains * np.array([REPEAT_DISCOUNT ** self.scan_counts.get((int(c) % self.W, int(c) // self.W), 0)
                                      for c in cand])
            for i, c in enumerate(cand):
                scored[("scan", int(c))] = (gains[i] / norm(bat[c], tim[c]), gains[i], dpos[c])

        bat, tim, ok = feasible("inspect")
        cand = np.nonzero(ok)[0]
        if len(cand):
            vals = inspect_value(b.mass3_all(), cand, FOUND_VALUE, MIN_INSPECT_MASS)
            for i, c in enumerate(cand):
                if vals[i] > 0:
                    scored[("inspect", int(c))] = (vals[i] / norm(bat[c], tim[c]), vals[i], dpos[c])

        bat, tim, ok = feasible("scan_area")
        cand = np.nonzero(ok)[0]
        clues_left = max(0, self.b["n_clues"] - len(self.clues))
        if len(cand) and clues_left:
            vals = clue_search_value(g, b.p, self.covered, area["radius"], area["detect_prob"],
                                     clues_left, self.b["clue_radius"], CLUE_GAIN, cand)
            for i, c in enumerate(cand):
                if vals[i] > 1e-6:
                    scored[("area", int(c))] = (vals[i] / norm(bat[c], tim[c]), vals[i], dpos[c])

        if not scored:
            return self._nothing_to_do()

        ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])
        best_key, best = ranked[0]
        if self.target in scored and scored[self.target][0] * 1.3 >= best[0]:
            best_key, best = self.target, scored[self.target]  # stick with the plan unless clearly beaten
        kind, idx = best_key
        self.target = best_key
        cell = (idx % self.W, idx // self.W)
        top = [{"kind": k, "cell": [i % self.W, i // self.W], "gain": round(float(v[1]), 3),
                "score": round(float(v[0]), 3), "steps": int(v[2])} for (k, i), v in ranked[:4]]

        strategy = {"scan": "localize", "area": "search-clues", "inspect": "approach-eve"}[kind]
        if idx == pos_i:
            self.target = None
            return self._act_here(kind, cell, best, strategy, top)
        d = self.step_toward([cell])
        if d is None:
            self.target = None
            return self._nothing_to_do()
        return Decision(A.move(d), strategy, self._why_going(kind, cell, best), top)

    # ----- explanations -----
    def _why_going(self, kind, cell, best):
        steps = int(best[2])
        if kind == "scan":
            return (f"Heading {steps} steps to {cell} to scan: expected to cut uncertainty by {best[1]:.2f} nats "
                    f"(entropy now {self.belief.entropy():.2f}).")
        if kind == "area":
            return f"Heading {steps} steps to {cell} to search for clues near where EVE is likely to be."
        return (f"Heading {steps} steps to {cell}: about {self.belief.mass_near(cell, 1):.0%} of the belief "
                f"sits within one cell of it, so inspecting there is worth the trip.")

    def _act_here(self, kind, cell, best, strategy, top):
        if kind == "scan":
            why = (f"Scanning here: expected to cut uncertainty by {best[1]:.2f} nats for little cost "
                   f"(entropy now {self.belief.entropy():.2f}).")
            return Decision(A.act(A.SCAN_SIGNAL), strategy, why, top)
        if kind == "area":
            return Decision(A.act(A.SCAN_AREA), strategy,
                            "Searching this area for clues: unscanned ground where belief says clues are likely.", top)
        m = self.belief.mass_near(cell, 1)
        return Decision(A.act(A.INSPECT), strategy,
                        f"Inspecting: {m:.0%} of the belief is within one cell of here.", top)

    def _nothing_to_do(self) -> Decision:
        if self.pos in self.stations:
            if self.battery < self.cap and self.time_left > self.tcosts["recharge"] + 3:
                return Decision(A.act(A.RECHARGE), "recharge",
                                "No affordable useful action from here; recharging to extend the search.")
            return Decision(A.act(A.FINISH), "finish",
                            "Safely at a station with nothing affordable left to try; ending the mission and reporting.")
        d = self.step_toward(self.stations, conservative=True) or self.step_toward(self.stations)
        if d:
            return Decision(A.move(d), "return", "Nothing affordable is worth the battery it would leave; heading to a station.")
        return Decision(A.act(A.FINISH), "finish", "No route to a station and no safe action; ending the mission.")
