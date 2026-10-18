"""Action vocabulary and per-action resource costs."""
from .grid import DIRS

MOVE, SCAN_SIGNAL, SCAN_AREA, INSPECT, RECHARGE, FINISH = (
    "MOVE", "SCAN_SIGNAL", "SCAN_AREA", "INSPECT", "RECHARGE", "FINISH",
)
KINDS = (MOVE, SCAN_SIGNAL, SCAN_AREA, INSPECT, RECHARGE, FINISH)

_COST_KEY = {MOVE: "move", SCAN_SIGNAL: "scan_signal", SCAN_AREA: "scan_area", INSPECT: "inspect"}


def move(direction: str) -> dict:
    return {"type": MOVE, "dir": direction}


def act(kind: str) -> dict:
    return {"type": kind}


def validate(action: dict) -> None:
    if not isinstance(action, dict) or action.get("type") not in KINDS:
        raise ValueError(f"Unknown action: {action!r}")
    if action["type"] == MOVE and action.get("dir") not in DIRS:
        raise ValueError(f"MOVE needs a direction N/E/S/W, got {action.get('dir')!r}")


def cost(resources: dict, action: dict) -> tuple[float, float]:
    """(battery, time) an action costs. Recharging costs time only."""
    t = action["type"]
    if t == RECHARGE:
        return 0.0, float(resources["time_costs"]["recharge"])
    if t == FINISH:
        return 0.0, 0.0
    key = _COST_KEY[t]
    return float(resources["costs"][key]), float(resources["time_costs"][key])
