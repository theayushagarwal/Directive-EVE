"""Optional: an LLM as the decision-maker, for the LLM-vs-classical comparison.

The model sees a text summary of the robot's state and belief and picks ONE action. It never sees
hidden truth. The universal battery guard in BaseAgent still applies, so a bad model answer
can waste resources but cannot strand the robot. Any parse failure falls back to a scan.
"""
import json
import os

import numpy as np

from . import actions as A
from .agent import BaseAgent, Decision
from .grid import DIRS

SYSTEM = (
    "You control WALL-E, a robot searching a grid for EVE using noisy signal scans. "
    "Reply with ONLY a JSON object: {\"action\": one of MOVE|SCAN_SIGNAL|SCAN_AREA|INSPECT|RECHARGE|FINISH, "
    "\"dir\": N|E|S|W (only for MOVE), \"why\": short reason}. "
    "INSPECT only finds EVE within 1 cell. Bearings are degrees, 0 = east, 90 = south (y grows downward)."
)


def groq_chat(system: str, user: str) -> str:
    import httpx

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"), "temperature": 0.2, "max_tokens": 120,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


class LLMAgent(BaseAgent):
    name = "llm"

    def __init__(self, *a, chat=groq_chat, **kw):
        super().__init__(*a, **kw)
        self.chat = chat
        self.failures = 0

    def _state_text(self) -> str:
        b = self.belief
        top = np.argsort(-b.p)[:5]
        cells = ", ".join(f"({i % self.W},{i // self.W}):{b.p[i]:.2f}" for i in top)
        rd = self.last_reading
        reading = "none yet" if rd is None else f"strength {rd['strength']:.2f}, bearing {rd['bearing_deg']:.0f} deg"
        return (f"Position {self.pos}; battery {self.battery:.0f}/{self.cap:.0f}; time left {self.time_left:.0f}.\n"
                f"Stations at {self.stations}. Move costs {self.costs['move']}, scan {self.costs['scan_signal']}, "
                f"area scan {self.costs['scan_area']}, inspect {self.costs['inspect']}.\n"
                f"Most likely EVE cells (cell:probability): {cells}. Uncertainty {b.entropy():.2f} nats.\n"
                f"Last signal reading (taken at the robot's last scan position): {reading}.\n"
                f"Clues found: {list(self.clues.values())}.")

    def choose(self) -> Decision:
        try:
            raw = self.chat(SYSTEM, self._state_text())
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            kind = data["action"]
            action = A.move(data["dir"]) if kind == A.MOVE else A.act(kind)
            A.validate(action)
            if action["type"] == A.MOVE:
                dx, dy = DIRS[action["dir"]]
                x, y = self.pos[0] + dx, self.pos[1] + dy
                if not (0 <= x < self.W and 0 <= y < self.H) or self.known[y, x] == 1:
                    raise ValueError("move into a known wall")
            return Decision(action, "llm", str(data.get("why", "LLM choice"))[:200])
        except Exception as exc:
            self.failures += 1
            return Decision(A.act(A.SCAN_SIGNAL), "llm", f"LLM answer unusable ({exc.__class__.__name__}); scanning instead.")
