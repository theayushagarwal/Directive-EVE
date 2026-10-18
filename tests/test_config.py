import json

import pytest
import yaml

from walle.config import ConfigError, load_config


def test_defaults_are_filled_in():
    cfg = load_config({"seed": 5})
    assert cfg["map"]["width"] == 21 and cfg["start"] == [1, 1] and cfg["resources"]["battery"] == 100


def test_nested_override_keeps_other_defaults():
    cfg = load_config({"resources": {"costs": {"move": 2}}})
    assert cfg["resources"]["costs"]["move"] == 2 and cfg["resources"]["costs"]["scan_signal"] == 2


@pytest.mark.parametrize("bad, msg", [
    ({"resources": {"battery": 0}}, "resources.battery"),
    ({"map": {"width": 3}}, "map.width"),
    ({"start": [99, 1]}, "outside"),
    ({"dropout": {"damaged": 1.5}}, "dropout.damaged"),
    ({"sensors": {"signal": {"d0": -1}}}, "d0"),
    ({"agent": {"safety_margin": 2}}, "safety_margin"),  # margin must beat the costliest action (3)
])
def test_bad_configs_give_readable_errors(bad, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(bad)


def test_yaml_and_json_files_load_identically(tmp_path):
    raw = {"name": "x", "seed": 9, "resources": {"battery": 80}}
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(raw))
    (tmp_path / "a.json").write_text(json.dumps(raw))
    assert load_config(tmp_path / "a.yaml") == load_config(tmp_path / "a.json")


def test_missing_file_and_wrong_extension(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ConfigError, match="must be"):
        load_config(tmp_path / "a.txt")


def test_shipped_sample_configs_load():
    from pathlib import Path

    for f in (Path(__file__).resolve().parent.parent / "configs").glob("mission_*.yaml"):
        load_config(f)
