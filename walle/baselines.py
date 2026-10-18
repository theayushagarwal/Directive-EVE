"""Simple comparison agents. They share the safety rules (so they never strand), and
differ only in how they spend spare resources."""
import math

from . import actions as A
from .agent import BaseAgent, Decision
from .grid import DIRS


class RandomAgent(BaseAgent):
    """Wanders, scans and inspects at random."""
    name = "random"

    def choose(self) -> Decision:
        r = self.rng.random()
        if r < 0.12:
            return Decision(A.act(A.INSPECT), "random", "Random choice: inspect here.")
        if r < 0.30:
            return Decision(A.act(A.SCAN_SIGNAL), "random", "Random choice: scan.")
        free = [d for d, (dx, dy) in DIRS.items()
                if 0 <= self.pos[0] + dx < self.W and 0 <= self.pos[1] + dy < self.H
                and self.known[self.pos[1] + dy, self.pos[0] + dx] != 1]
        d = free[int(self.rng.integers(len(free)))] if free else "N"
        return Decision(A.move(d), "random", f"Random choice: move {d}.")


class GradientAgent(BaseAgent):
    """Reactive: scan, walk a few steps along the last bearing, repeat; inspect when the
    signal is strong. No memory of past readings (the fair 'classical' baseline)."""
    name = "gradient"
    WALK = 3

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.walk_left = 0
        self.heading = None
        self.inspects_here = 0
        self.last_pos = None

    def _dir_from_bearing(self, bearing_deg):
        a = math.radians(bearing_deg)
        dx, dy = math.cos(a), math.sin(a)
        order = sorted(DIRS.items(), key=lambda kv: -(kv[1][0] * dx + kv[1][1] * dy))
        for d, (ddx, ddy) in order:
            x, y = self.pos[0] + ddx, self.pos[1] + ddy
            if 0 <= x < self.W and 0 <= y < self.H and self.known[y, x] != 1:
                return d
        return None

    def choose(self) -> Decision:
        if self.pos != self.last_pos:
            self.inspects_here, self.last_pos = 0, self.pos
        strong = 0.8 * self.b["sensors"]["signal"]["S0"]
        r = self.last_reading
        if r and r["strength"] > strong and self.inspects_here < 2:
            self.inspects_here += 1
            return Decision(A.act(A.INSPECT), "approach-eve", "Signal is strong: inspecting.")
        if self.walk_left > 0 and r:
            d = self._dir_from_bearing(r["bearing_deg"])
            if d:
                self.walk_left -= 1
                return Decision(A.move(d), "localize", f"Following the last bearing ({r['bearing_deg']:.0f} deg).")
        self.walk_left = self.WALK
        self.last_reading = None if self.last_reading is None else self.last_reading
        return Decision(A.act(A.SCAN_SIGNAL), "localize", "Taking a fresh signal reading.")
