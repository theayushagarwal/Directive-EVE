import json
import sys
from pathlib import Path

import numpy as np

from walle.runner import run_mission
from walle.scenarios import make_config
from walle.trace import load_jsonl

REQUIRED = {"outcome", "inferred_location", "location_confidence", "inferred_status", "evidence",
            "recommended_next_step", "resources", "strategy_timeline", "evaluation_only"}


def test_run_writes_every_deliverable(tmp_path):
    r = run_mission(make_config(3, "decoys"), out_dir=tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"trace.jsonl", "report.json", "report.md", "belief.npz", "map.txt", "mission.png", "visualizer.html"} <= names
    assert (tmp_path / "mission.png").stat().st_size > 5000
    assert (tmp_path / "visualizer.html").stat().st_size > 10000
    assert np.load(tmp_path / "belief.npz")["belief"].shape[0] == r.steps + 1


def test_trace_logs_action_observation_decision_and_reason(tmp_path):
    r = run_mission(make_config(1, "clean"), out_dir=tmp_path)
    recs = load_jsonl(tmp_path / "trace.jsonl")
    assert len(recs) == r.steps
    for rec in recs:
        assert rec["action"]["type"] and rec["reason"] and rec["strategy"]
        assert {"pos", "battery", "time_left"} <= set(rec["before"])
        assert {"entropy", "map_cell", "status_p"} <= set(rec["belief"])
        assert "event" in rec["observation"]
    assert any(rec["candidates"] for rec in recs)  # the alternatives it weighed are logged too


def test_report_has_every_required_section_and_is_valid_json(tmp_path):
    run_mission(make_config(2, "clean"), out_dir=tmp_path)
    rep = json.loads((tmp_path / "report.json").read_text())
    assert REQUIRED <= set(rep)
    assert 0 <= rep["location_confidence"] <= 1
    assert rep["inferred_status"]["status"] in ("active", "damaged", "dormant")
    md = (tmp_path / "report.md").read_text()
    for heading in ("Where is EVE", "Recommended next step", "Evidence", "How the strategy changed", "Ground truth"):
        assert heading in md


def test_an_unfinished_search_recommends_a_concrete_next_step(tmp_path):
    cfg = make_config(4, "large")
    cfg["resources"] = {"time": 25}  # far too little time to finish
    r = run_mission(cfg, out_dir=tmp_path)
    assert not r.found
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["inferred_location"]["confirmed"] is False
    nxt = rep["recommended_next_step"]
    assert "inspect" in nxt.lower() or "readings" in nxt.lower()


def test_ascii_view_shows_robot_station_and_belief():
    from walle import viz

    r = run_mission(make_config(1, "clean"))
    text = viz.ascii_report(r)
    assert "R" in text and "S" in text and "BELIEF" in text


def test_script_runs_end_to_end(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parent.parent
    out = subprocess.run([sys.executable, str(root / "scripts" / "run_mission.py"),
                          str(root / "configs" / "mission_easy.yaml"), "--out", str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "mission_easy" in out.stdout and (tmp_path / "report.md").exists()
