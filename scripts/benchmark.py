"""Generalisation benchmark over many random missions.

  python scripts/benchmark.py --seeds 20 --out runs/benchmark

Every agent plays the same missions (same seeds and scenarios). Outcomes are reported with
a 95% Wilson interval, because a success rate from a few dozen runs is a noisy number.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from walle.agent import BeliefAgent  # noqa: E402
from walle.baselines import GradientAgent, RandomAgent  # noqa: E402
from walle.runner import run_mission  # noqa: E402
from walle.scenarios import SCENARIOS, make_config  # noqa: E402

AGENTS = {"belief": BeliefAgent, "gradient": GradientAgent, "random": RandomAgent}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--first-seed", type=int, default=1000)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--agents", default="belief,gradient,random")
    ap.add_argument("--out", default="runs/benchmark")
    a = ap.parse_args()
    scenarios, agents = a.scenarios.split(","), a.agents.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "results.csv"
    rows = []
    if csv_path.exists():  # resume: keep finished work, skip it below
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                rows.append({**r, "seed": int(r["seed"]), "found": int(r["found"]), "steps": int(r["steps"]),
                             "battery": float(r["battery"]), "time_used": float(r["time_used"]),
                             "recharges": int(r["recharges"]), "guess_error": float(r["guess_error"])})
    done = {(r["scenario"], r["agent"], r["seed"]) for r in rows}

    def save():
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scenario", "agent", "seed", "outcome", "found", "steps", "battery",
                                              "time_used", "recharges", "guess_error"])
            w.writeheader()
            w.writerows(rows)

    for sc in scenarios:
        for name in agents:
            t0 = time.time()
            for s in range(a.first_seed, a.first_seed + a.seeds):
                if (sc, name, s) in done:
                    continue
                r = run_mission(make_config(s, sc), AGENTS[name])
                rows.append(dict(scenario=sc, agent=name, seed=s, outcome=r.outcome, found=int(r.found),
                                 steps=r.steps, battery=round(r.energy_spent, 1), time_used=round(r.time_used, 1),
                                 recharges=r.recharges, guess_error=round(r.map_error, 2)))
            save()
            print(f"{sc:7s} {name:9s} ready ({time.time() - t0:.0f}s this run)", flush=True)

    lines = ["| scenario | agent | found | 95% interval | stranded | mean steps (found) | mean battery | mean guess error |",
             "|---|---|---|---|---|---|---|---|"]
    for sc in scenarios:
        for name in agents:
            rs = [r for r in rows if r["scenario"] == sc and r["agent"] == name]
            k, n = sum(r["found"] for r in rs), len(rs)
            lo, hi = wilson(k, n)
            fs = [r["steps"] for r in rs if r["found"]]
            lines.append(f"| {sc} | {name} | {k}/{n} | {lo:.0%} to {hi:.0%} | {sum(r['outcome'] == 'stranded' for r in rs)} | "
                         f"{(sum(fs) / len(fs)) if fs else float('nan'):.0f} | {sum(r['battery'] for r in rs) / n:.0f} | "
                         f"{sum(r['guess_error'] for r in rs) / n:.1f} |")
    table = "\n".join(lines)
    (out / "summary.md").write_text(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    main()
