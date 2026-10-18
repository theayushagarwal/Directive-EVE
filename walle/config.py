"""Mission config: load JSON/YAML, merge with defaults, validate with readable errors."""
import copy
import json
from pathlib import Path


class ConfigError(ValueError):
    pass


DEFAULTS = {
    "name": "mission",
    "seed": 0,
    "map": {"width": 21, "height": 21, "wall_density": 0.12, "walls": None},
    "start": None,
    "stations": None,
    "resources": {
        "battery": 100,
        "time": 160,
        "recharge_amount": 30,
        "costs": {"move": 1, "scan_signal": 2, "scan_area": 3, "inspect": 1},
        "time_costs": {"move": 1, "scan_signal": 1, "scan_area": 2, "inspect": 1, "recharge": 3},
    },
    "sensors": {
        "signal": {"S0": 1.0, "d0": 5.0, "sigma_log": 0.25, "sigma_bearing_deg": 18.0, "floor": 0.005},
        "area": {"radius": 3.0, "detect_prob": 0.9},
    },
    "initial_signal": None,
    "dropout": {"active": 0.05, "damaged": 0.35, "dormant": 0.75},
    "hidden": {
        "eve": None,
        "clues": None,
        "decoys": None,
        "n_clues": 4,
        "n_decoys": 0,
        "decoy_clue_fraction": 0.25,
        "clue_sigma": 2.0,
        "clue_radius": 9,
        "spoof_prob": 0.0,
        "eve_dist_range": None,
    },
    "agent": {"safety_margin": 4, "spoof_epsilon": 0.15, "clue_reliability": 0.7},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _num(cfg, path, lo=None, hi=None, positive=False):
    cur = cfg
    for p in path.split("."):
        cur = cur[p]
    if isinstance(cur, bool) or not isinstance(cur, (int, float)):
        raise ConfigError(f"{path} must be a number, got {cur!r}")
    if positive and cur <= 0:
        raise ConfigError(f"{path} must be greater than 0, got {cur}")
    if lo is not None and cur < lo:
        raise ConfigError(f"{path} must be at least {lo}, got {cur}")
    if hi is not None and cur > hi:
        raise ConfigError(f"{path} must be at most {hi}, got {cur}")


def validate(cfg: dict) -> dict:
    m = cfg["map"]
    _num(cfg, "map.width", 7, 60)
    _num(cfg, "map.height", 7, 60)
    _num(cfg, "map.wall_density", 0, 0.45)
    W, H = int(m["width"]), int(m["height"])
    if cfg["start"] is None:
        cfg["start"] = [1, 1]
    sx, sy = cfg["start"]
    if not (0 <= sx < W and 0 <= sy < H):
        raise ConfigError(f"start {cfg['start']} is outside the {W}x{H} map")
    for st in cfg["stations"] or []:
        if not (0 <= st[0] < W and 0 <= st[1] < H):
            raise ConfigError(f"station {st} is outside the map")
    _num(cfg, "resources.battery", positive=True)
    _num(cfg, "resources.time", positive=True)
    _num(cfg, "resources.recharge_amount", positive=True)
    for k, v in cfg["resources"]["costs"].items():
        if not isinstance(v, (int, float)) or v < 0:
            raise ConfigError(f"resources.costs.{k} must be a non-negative number")
    if cfg["resources"]["costs"]["move"] <= 0:
        raise ConfigError("resources.costs.move must be greater than 0")
    for k, v in cfg["resources"]["time_costs"].items():
        if not isinstance(v, (int, float)) or v <= 0:
            raise ConfigError(f"resources.time_costs.{k} must be greater than 0")
    for k in ("S0", "d0", "sigma_log", "sigma_bearing_deg", "floor"):
        _num(cfg, f"sensors.signal.{k}", positive=True)
    _num(cfg, "sensors.area.radius", positive=True)
    _num(cfg, "sensors.area.detect_prob", 0, 1)
    for k in ("active", "damaged", "dormant"):
        _num(cfg, f"dropout.{k}", 0, 1)
    _num(cfg, "hidden.spoof_prob", 0, 1)
    _num(cfg, "hidden.n_clues", 0, 50)
    _num(cfg, "hidden.n_decoys", 0, 50)
    _num(cfg, "agent.safety_margin", 0)
    costliest = max(cfg["resources"]["costs"].values())
    if cfg["agent"]["safety_margin"] < costliest + 1:
        raise ConfigError(
            f"agent.safety_margin ({cfg['agent']['safety_margin']}) must exceed the costliest action "
            f"({costliest}) by at least 1, or the robot can run dry on its way home")
    _num(cfg, "agent.spoof_epsilon", 0, 0.9)
    _num(cfg, "agent.clue_reliability", 0, 1)
    return cfg


def load_config(src) -> dict:
    """Accept a dict or a path to a .json/.yaml/.yml file."""
    if isinstance(src, dict):
        raw = src
    else:
        path = Path(src)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            raw = yaml.safe_load(text) or {}
        elif path.suffix.lower() == ".json":
            raw = json.loads(text)
        else:
            raise ConfigError("Config must be .json, .yaml or .yml")
    return validate(_merge(DEFAULTS, raw))
