"""Run one mission:  python scripts/run_mission.py configs/mission_easy.yaml --out runs/mission_easy"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from walle.agent import BeliefAgent  # noqa: E402
from walle.baselines import GradientAgent, RandomAgent  # noqa: E402
from walle.config import ConfigError, load_config  # noqa: E402
from walle.runner import run_mission  # noqa: E402

AGENTS = {"belief": BeliefAgent, "gradient": GradientAgent, "random": RandomAgent}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--agent", default="belief", choices=[*AGENTS, "llm"])
    ap.add_argument("--out", help="folder for trace, report and images")
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF")
    ap.add_argument("--seed", type=int, help="override the config seed")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        cfg = load_config(a.config)
    except ConfigError as e:
        sys.exit(f"Config problem: {e}")
    if a.seed is not None:
        cfg["seed"] = a.seed
    if a.agent == "llm":
        from walle.llm_agent import LLMAgent

        cls = LLMAgent
    else:
        cls = AGENTS[a.agent]
    res = run_mission(cfg, cls, out_dir=a.out, gif=a.gif)
    if not a.quiet:
        mp = res.agent.belief.map_cell()
        print(f"{cfg['name']}: {res.outcome} in {res.steps} steps | battery spent {res.energy_spent:.0f} | "
              f"time {res.time_used:.0f}/{cfg['resources']['time']} | recharges {res.recharges} | "
              f"best guess {mp}, off by {res.map_error:.1f} cells")
        if a.out:
            print(f"Wrote trace, report and images to {a.out}")


if __name__ == "__main__":
    main()
