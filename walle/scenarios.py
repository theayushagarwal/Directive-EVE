"""Named mission generators used by the benchmark (and handy for experiments)."""

SCENARIOS = {
    "clean": {},
    "noisy": {
        "sensors": {"signal": {"sigma_log": 0.5, "sigma_bearing_deg": 35.0}},
        "dropout": {"active": 0.15, "damaged": 0.45, "dormant": 0.8},
    },
    "decoys": {
        "sensors": {"signal": {"sigma_log": 0.35, "sigma_bearing_deg": 25.0}},
        "hidden": {"n_decoys": 3, "spoof_prob": 0.3},
    },
    "large": {
        "map": {"width": 31, "height": 31, "wall_density": 0.18},
        "resources": {"battery": 130, "time": 260},
        "sensors": {"signal": {"sigma_log": 0.35, "sigma_bearing_deg": 25.0}},
        "hidden": {"n_decoys": 2, "spoof_prob": 0.15, "clue_radius": 12},
    },
}


def make_config(seed: int, scenario: str = "clean") -> dict:
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    from .config import _merge

    return _merge({"name": f"{scenario}-{seed}", "seed": seed}, SCENARIOS[scenario])
