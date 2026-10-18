import ast
from pathlib import Path

import numpy as np
import pytest

from walle.agent import BeliefAgent
from walle.baselines import GradientAgent, RandomAgent
from walle.runner import run_mission
from walle.scenarios import SCENARIOS, make_config

PKG = Path(__file__).resolve().parent.parent / "walle"


def test_agents_never_import_the_world_or_the_environment():
    """The agent can only learn clue and EVE locations through observations."""
    for name in ("agent", "baselines", "belief", "planner", "llm_agent", "sensors"):
        tree = ast.parse((PKG / f"{name}.py").read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                assert m.split(".")[-1] not in ("world", "env"), f"{name}.py imports {m}"


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("agent_cls", [BeliefAgent, GradientAgent, RandomAgent])
def test_no_agent_is_ever_stranded_or_overdrawn(scenario, agent_cls):
    for seed in range(4):
        r = run_mission(make_config(seed, scenario), agent_cls)
        assert r.outcome != "stranded", f"{agent_cls.name} stranded on {scenario}/{seed}"
        assert min(r.battery) >= 0
        for rec in r.trace.records:
            assert rec["observation"]["battery"] >= 0


def test_belief_stays_normalised_throughout_a_mission():
    r = run_mission(make_config(2, "decoys"))
    for snap in r.snapshots:
        assert abs(float(snap.sum()) - 1) < 1e-3


def test_belief_agent_finds_eve_on_clean_missions_and_beats_random():
    rs = [run_mission(make_config(s, "clean")) for s in range(8)]
    rnd = [run_mission(make_config(s, "clean"), RandomAgent) for s in range(8)]
    assert sum(r.found for r in rs) >= 7
    assert sum(r.found for r in rs) > sum(r.found for r in rnd)


def test_turns_back_when_battery_cannot_cover_the_trip_home():
    cfg = make_config(1, "clean")
    cfg["map"] = {"width": 21, "height": 21, "wall_density": 0}
    cfg["stations"] = [[1, 1]]
    from walle.config import load_config
    from walle.env import Environment

    env = Environment(load_config(cfg))
    a = BeliefAgent(env.briefing())
    for x in range(1, 12):  # a corridor of known-free cells, 10 steps from the station
        a.known[1, x] = 0
    a.pos, a.battery = (11, 1), 13.0  # needs 10 + margin 4 = 14
    d = a.decide()
    assert d.strategy == "return" and d.action == {"type": "MOVE", "dir": "W"}
    assert "turning back" in d.reason.lower()


def test_recharges_at_a_station_when_low():
    from walle.config import load_config
    from walle.env import Environment

    a = BeliefAgent(Environment(load_config({"seed": 1})).briefing())
    a.battery = 20.0
    assert a.decide().strategy == "recharge"


def test_guard_refuses_actions_that_would_strand_a_baseline():
    from walle.config import load_config
    from walle.env import Environment

    a = RandomAgent(Environment(load_config({"seed": 1, "stations": [[1, 1]]})).briefing())
    for x in range(1, 9):
        a.known[1, x] = 0
    a.pos, a.battery = (8, 1), 7.0  # exactly enough to walk home, nothing to spare
    from walle.agent import Decision
    from walle import actions as A

    d = a.guard(Decision(A.act(A.SCAN_SIGNAL), "random", "x"))
    assert d.strategy == "return"


def test_same_seed_gives_identical_missions():
    a = run_mission(make_config(5, "decoys"))
    b = run_mission(make_config(5, "decoys"))
    assert [r["action"] for r in a.trace.records] == [r["action"] for r in b.trace.records]


def test_explicit_config_reproduces_the_generated_world():
    from walle.config import load_config
    from walle.world import World
    import importlib.util

    spec = importlib.util.spec_from_file_location("mk", PKG.parent / "scripts" / "make_configs.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    cfg = load_config(make_config(3, "decoys"))
    explicit = load_config(mk.explicit(cfg))
    w1, w2 = World(cfg), World(explicit)
    assert (w1.walls == w2.walls).all() and w1.eve_pos == w2.eve_pos and w1.decoys == w2.decoys
    assert [c["target"] for c in w1.clues] == [c["target"] for c in w2.clues]
