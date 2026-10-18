import json

import pytest

from walle import actions as A
from walle.config import load_config
from walle.env import Environment


def open_cfg(**over):
    """An empty 15x15 arena, EVE at (10,10), station at the start."""
    cfg = {"seed": 1, "map": {"width": 15, "height": 15, "wall_density": 0},
           "start": [1, 1], "stations": [[1, 1]],
           "hidden": {"eve": {"pos": [10, 10], "status": "active"}, "n_clues": 0}}
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return load_config(cfg)


def test_no_hidden_truth_leaks_through_briefing_or_observations():
    env = Environment(open_cfg())
    blob = json.dumps(env.briefing()) + json.dumps(env.step(A.act(A.SCAN_SIGNAL)), default=str)
    for secret in ("hidden", "eve_pos", "ground_truth", "decoy", '"eve"'):
        assert secret not in blob
    assert "[10, 10]" not in blob


def test_costs_are_charged_exactly():
    env = Environment(open_cfg())
    o = env.step(A.act(A.SCAN_SIGNAL))
    assert o["battery"] == 98 and o["time_left"] == 159
    o = env.step(A.move("E"))
    assert o["battery"] == 97 and o["pos"] == [2, 1]
    o = env.step(A.act(A.SCAN_AREA))
    assert o["battery"] == 94 and o["time_left"] == 160 - 1 - 1 - 2


def test_bumping_into_the_edge_costs_but_does_not_move():
    env = Environment(open_cfg())
    o = env.step(A.move("N"))  # (1,1) -> (1,0) is fine
    o = env.step(A.move("N"))  # off the map
    assert o["event"] == "bumped" and o["pos"] == [1, 0] and o["battery"] == 98


def test_recharge_only_works_at_a_station():
    env = Environment(open_cfg())
    env.step(A.move("E"))
    assert env.step(A.act(A.RECHARGE))["event"] == "not_at_station"
    env.step(A.move("W"))
    o = env.step(A.act(A.RECHARGE))
    assert o["event"] == "recharged" and o["battery"] == 100  # 98 + 30, capped at capacity


def test_inspect_only_finds_eve_within_one_cell():
    env = Environment(open_cfg(resources={"battery": 200, "time": 400}))
    for _ in range(8):
        env.step(A.move("E"))
    for _ in range(8):
        env.step(A.move("S"))
    o = env.step(A.act(A.INSPECT))  # at (9,9): diagonally adjacent to EVE at (10,10)
    assert o["found"] and o["status"] == "active" and o["done"] and o["outcome"] == "found"


def test_inspect_far_away_finds_nothing():
    env = Environment(open_cfg())
    o = env.step(A.act(A.INSPECT))
    assert o["found"] is False and not o["done"]


def test_running_dry_away_from_a_station_strands_the_robot():
    env = Environment(open_cfg(resources={"battery": 3}))
    outcome = None
    for _ in range(3):
        o = env.step(A.move("E"))
        outcome = o["outcome"]
    assert outcome == "stranded"


def test_empty_battery_at_a_station_is_not_stranded():
    env = Environment(open_cfg(resources={"battery": 2, "costs": {"scan_signal": 2}}))
    o = env.step(A.act(A.SCAN_SIGNAL))
    assert o["battery"] == 0 and o["outcome"] is None


def test_timeout_ends_the_mission():
    env = Environment(open_cfg(resources={"time": 2}))
    env.step(A.act(A.SCAN_SIGNAL))
    o = env.step(A.act(A.SCAN_SIGNAL))
    assert o["done"] and o["outcome"] == "timeout"


def test_refused_action_still_burns_time_and_never_goes_negative():
    env = Environment(open_cfg(resources={"battery": 2}))
    env.step(A.move("E"))  # 1 left, away from the station
    t = env.time_left
    o = env.step(A.act(A.SCAN_AREA))  # costs 3, have 1
    assert o["event"] == "insufficient_battery" and not o["ok"]
    assert o["battery"] == 1 and env.time_left == t - 1


def test_same_config_same_mission():
    def run():
        env = Environment(open_cfg(hidden={"spoof_prob": 0.0}))
        return [env.step(A.act(A.SCAN_SIGNAL)).get("reading") for _ in range(5)]

    assert run() == run()


def test_actions_are_validated():
    env = Environment(open_cfg())
    with pytest.raises(ValueError):
        env.step({"type": "TELEPORT"})
    with pytest.raises(ValueError):
        env.step({"type": "MOVE", "dir": "UP"})
