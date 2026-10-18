"""Full action trace: every action, observation, decision, reason and belief summary."""
import json

import numpy as np


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not serialisable: {type(o)}")


class Trace:
    def __init__(self):
        self.records: list[dict] = []

    def add(self, step, before: dict, decision, obs: dict, agent) -> dict:
        b = agent.belief
        mp = b.map_cell()
        status, status_p = b.status_estimate()
        rec = {
            "step": step,
            "before": before,
            "strategy": decision.strategy,
            "reason": decision.reason,
            "candidates": decision.candidates,
            "action": decision.action,
            "observation": {k: v for k, v in obs.items() if k not in ("action", "step")},
            "belief": {
                "entropy": round(b.entropy(), 4),
                "map_cell": list(mp),
                "map_mass_r2": round(b.mass_near(mp, 2), 4),
                "status": status,
                "status_p": {k: round(v, 4) for k, v in b.status_p.items()},
            },
        }
        self.records.append(rec)
        return rec

    def write_jsonl(self, path) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r, default=_default) + "\n")


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
