"""One-click demonstration script for judges and evaluators.

Runs both benchmark missions, generates all deliverables, and opens the
interactive visualizer dashboard in your default web browser.

Usage:
  python scripts/demo.py
  python scripts/demo.py --no-browser
"""
import argparse
import sys
import time
import webbrowser
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from walle.agent import BeliefAgent  # noqa: E402
from walle.config import load_config  # noqa: E402
from walle.runner import run_mission  # noqa: E402


def banner(text: str):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser")
    args = parser.parse_args()

    banner("WALL-E & EVE's Signal: Autonomous Search & Rescue Demonstration")
    print("Initializing environment and executing benchmark missions...\n")

    # 1. Run mission_easy
    print("[1/2] Running 'mission_easy' (Clean Sensors, Baseline Triangulation)...")
    t0 = time.time()
    cfg_easy = load_config(ROOT / "configs" / "mission_easy.yaml")
    res_easy = run_mission(cfg_easy, BeliefAgent, out_dir=ROOT / "runs" / "mission_easy", gif=True)
    dt_easy = time.time() - t0
    print(f"      Status: {res_easy.outcome.upper()} in {res_easy.steps} steps ({dt_easy:.1f}s)")
    print(f"      Battery spent: {res_easy.energy_spent:.0f} | Time used: {res_easy.time_used:.0f}/{cfg_easy['resources']['time']}")
    print(f"      Best guess: {res_easy.agent.belief.map_cell()} | Error: {res_easy.map_error:.2f} cells\n")

    # 2. Run mission_decoys
    print("[2/2] Running 'mission_decoys' (3 Decoys, 30% Spoofing, Damaged EVE, Recharge Cycle)...")
    t0 = time.time()
    cfg_decoys = load_config(ROOT / "configs" / "mission_decoys.yaml")
    res_decoys = run_mission(cfg_decoys, BeliefAgent, out_dir=ROOT / "runs" / "mission_decoys", gif=True)
    dt_decoys = time.time() - t0
    print(f"      Status: {res_decoys.outcome.upper()} in {res_decoys.steps} steps ({dt_decoys:.1f}s)")
    print(f"      Battery spent: {res_decoys.energy_spent:.0f} (Recharges: {res_decoys.recharges})")
    print(f"      Best guess: {res_decoys.agent.belief.map_cell()} | Error: {res_decoys.map_error:.2f} cells")

    banner("All Deliverables Generated Successfully")
    print("Each run output folder contains:")
    print("  - visualizer.html : Interactive Web UI dashboard (heatmap, scrubber, HUD)")
    print("  - trace.jsonl     : Step-by-step actions, perceptions, and natural language reasoning")
    print("  - report.md/json  : Inferred location, status confidence, and next-step plan")
    print("  - mission.png/gif : Colored path, timeline charts, and animated replay")
    print("  - map.txt         : ASCII view of map, terrain, and belief grid")
    print("  - belief.npz      : Exact probability grid tensor at every step")

    # Launch browser
    html_target = (ROOT / "runs" / "mission_decoys" / "visualizer.html").resolve()
    if not args.no_browser:
        banner("Opening Interactive Web Visualizer in Browser")
        print(f"Target: {html_target.as_uri()}")
        webbrowser.open(html_target.as_uri())

    banner("Quick Commands for Judges & Evaluators")
    print("  - Run test suite (all 63 tests)    : python -m pytest -q")
    print("  - Launch visualizer for any run    : python scripts/visualize.py runs/mission_decoys")
    print("  - Run 240-mission generalization bm: python scripts/benchmark.py --seeds 20\n")


if __name__ == "__main__":
    main()
