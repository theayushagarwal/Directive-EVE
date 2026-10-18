import importlib.util
import json
from pathlib import Path

from walle import actions as A
from walle.config import load_config
from walle.env import Environment
from walle.llm_agent import LLMAgent

ROOT = Path(__file__).resolve().parent.parent


def agent(reply):
    briefing = Environment(load_config({"seed": 1, "stations": [[1, 1]]})).briefing()
    return LLMAgent(briefing, chat=lambda system, user: reply)


def test_valid_llm_answer_is_used():
    d = agent(json.dumps({"action": "SCAN_SIGNAL", "why": "need data"})).decide()
    assert d.action == A.act(A.SCAN_SIGNAL) and d.reason == "need data"


def test_garbage_llm_answer_falls_back_to_a_scan():
    a = agent("I think we should go left!")
    d = a.decide()
    assert d.action == A.act(A.SCAN_SIGNAL) and a.failures == 1


def test_llm_cannot_walk_the_robot_into_a_known_wall_or_off_the_map():
    d = agent(json.dumps({"action": "MOVE", "dir": "N"})).decide()  # (1,1) -> (1,0) is fine
    assert d.action["type"] == "MOVE"
    a = agent(json.dumps({"action": "MOVE", "dir": "W"}))
    a.known[1, 0] = 1
    assert a.decide().action == A.act(A.SCAN_SIGNAL)


def test_llm_cannot_strand_the_robot_the_guard_still_applies():
    a = agent(json.dumps({"action": "MOVE", "dir": "E"}))
    for x in range(1, 9):
        a.known[1, x] = 0
    a.pos, a.battery = (8, 1), 7.0
    assert a.decide().strategy == "return"


def test_wilson_interval_behaves():
    spec = importlib.util.spec_from_file_location("bm", ROOT / "scripts" / "benchmark.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    lo, hi = bm.wilson(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
    assert bm.wilson(0, 0) == (0.0, 0.0)
    lo2, hi2 = bm.wilson(90, 100)
    assert (hi2 - lo2) < (hi - lo)  # more runs, tighter interval
