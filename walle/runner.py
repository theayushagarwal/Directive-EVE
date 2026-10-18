"""Run one mission end to end and (optionally) write trace, report and visuals."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agent import BeliefAgent
from .config import load_config
from .env import Environment
from .trace import Trace


@dataclass
class RunResult:
    outcome: str
    steps: int
    energy_spent: float
    time_used: float
    recharges: int
    found: bool
    map_error: float          # distance from the agent's best guess to the truth (evaluation only)
    env: Environment
    agent: object
    trace: Trace
    path: list = field(default_factory=list)
    strategies: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    entropy: list = field(default_factory=list)
    battery: list = field(default_factory=list)


def run_mission(cfg, agent_cls=BeliefAgent, out_dir=None, max_steps: int = 800, gif: bool = False,
                agent_kwargs: dict | None = None) -> RunResult:
    cfg = load_config(cfg)
    env = Environment(cfg)
    agent = agent_cls(env.briefing(), seed=int(cfg["seed"]), **(agent_kwargs or {}))
    trace = Trace()
    path, strategies = [tuple(agent.pos)], []
    snaps = [agent.belief.snapshot()]
    ent, bat = [agent.belief.entropy()], [agent.battery]
    energy = 0.0
    outcome = "step_limit"

    for step in range(1, max_steps + 1):
        dec = agent.decide()
        before = {"pos": list(agent.pos), "battery": agent.battery, "time_left": agent.time_left}
        obs = env.step(dec.action)
        energy += max(0.0, before["battery"] - obs["battery"])
        agent.observe(obs)
        trace.add(step, before, dec, obs, agent)
        path.append(tuple(agent.pos))
        strategies.append(dec.strategy)
        snaps.append(agent.belief.snapshot())
        ent.append(agent.belief.entropy())
        bat.append(agent.battery)
        if obs["done"]:
            outcome = obs["outcome"]
            break

    truth = env.ground_truth()
    # Once EVE is confirmed, her location is known to within 1 cell of where the robot stands.
    mp = tuple(agent.pos) if agent.found else agent.belief.map_cell()
    err = float(np.hypot(mp[0] - truth["eve_pos"][0], mp[1] - truth["eve_pos"][1]))
    res = RunResult(outcome, len(trace.records), energy, cfg["resources"]["time"] - env.time_left,
                    agent.stats["recharges"], agent.found, err, env, agent, trace, path, strategies,
                    snaps, ent, bat)

    if out_dir:
        from .report import write_all

        write_all(res, cfg, Path(out_dir), gif=gif)
    return res
