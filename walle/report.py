"""Final mission report: evidence found, inferred location/status, confidence, next step."""
import json
from pathlib import Path

import numpy as np

from .planner import expected_info_gain
from .trace import _default

RECOVERY = {
    "active": "EVE is responsive: open a comms link and escort her to the nearest station.",
    "damaged": "EVE is damaged: bring a repair kit and stabilise her before moving her.",
    "dormant": "EVE is dormant: bring a wake-up beacon or a tow, and do not expect her to move on her own.",
}


def _segments(strategies):
    segs, start = [], 0
    for i in range(1, len(strategies) + 1):
        if i == len(strategies) or strategies[i] != strategies[start]:
            segs.append({"phase": strategies[start], "from_step": start + 1, "to_step": i})
            start = i
    return segs


def _readings(trace):
    out = []
    for r in trace.records:
        o = r["observation"]
        if r["action"]["type"] == "SCAN_SIGNAL" and o.get("reading"):
            out.append({"step": r["step"], "pos": o["pos"], **o["reading"]})
    return out


def _vantage(agent):
    """Best place to take more readings from, ignoring battery (for the recommendation)."""
    g, b = agent.belief.g, agent.belief
    cand = np.nonzero((agent.known.ravel() != 1))[0]
    gains = expected_info_gain(g, b.p, agent.b["sensors"]["signal"], b.eps, cand, np.random.default_rng(0))
    best = int(cand[int(np.argmax(gains))])
    return [best % agent.W, best // agent.W], float(gains.max())


def build_report(res, cfg: dict) -> dict:
    agent, env, trace = res.agent, res.env, res.trace
    b = agent.belief
    mp = b.map_cell()
    status, status_p = b.status_estimate()
    d_station = agent.dist_to_station()
    readings = _readings(trace)
    mv, margin = agent.costs["move"], agent.params["safety_margin"]

    if agent.found:
        loc = {"cell": list(agent.pos), "within_cells": 1, "confirmed": True}
        confidence = 1.0
        status, status_p = agent.found_status, 1.0
        status_note = "confirmed by direct inspection"
    else:
        loc = {"cell": list(mp), "within_cells": 2, "confirmed": False,
               "centroid": [round(c, 2) for c in b.centroid()],
               "credible_region_80pct_cells": len(b.credible_cells(0.8))}
        confidence = round(b.mass_near(mp, 2), 3)
        status_note = (f"inferred from {agent.stats['dropouts']} signal dropouts in {agent.stats['scans']} scans; "
                       "a damaged or dormant EVE drops out more often")

    clues = []
    for c in agent.clues.values():
        near = b.mass_near(tuple(c["target"]), 2)
        clues.append({"id": c["id"], "found_at": c["pos"], "points_to": c["target"], "spread": c["sigma"],
                      "assessment": ("consistent with the rest of the evidence" if near >= 0.05 else
                                     "conflicts with the other evidence (possibly a decoy)")})
    strongest = max(readings, key=lambda r: r["strength"]) if readings else None
    inspected = [r["observation"]["pos"] for r in trace.records
                 if r["action"]["type"] == "INSPECT" and not r["observation"].get("found")]

    # next step
    anchor = loc["cell"]
    home_steps = d_station[anchor[1], anchor[0]]
    need = (home_steps * 2 * mv + margin) if np.isfinite(home_steps) else None
    if agent.found:
        nxt = RECOVERY.get(status, "Recover EVE.")
        if need:
            nxt += f" She is {home_steps:.0f} steps from a station; a round trip needs about {need:.0f} battery."
    elif confidence >= 0.5:
        nxt = (f"Return to a station, recharge fully, then inspect around {anchor} first "
               f"({confidence:.0%} of the belief is within 2 cells of it)")
        nxt += f". A round trip from the nearest station needs about {need:.0f} battery." if need else "."
    else:
        v, gain = _vantage(agent)
        nxt = (f"Evidence is still diffuse (only {confidence:.0%} within 2 cells of the best guess {anchor}). "
               f"Do not commit to an inspection yet: take more signal readings from {v}, which should remove "
               f"about {gain:.1f} nats of uncertainty, then re-plan.")

    truth = env.ground_truth()
    return {
        "mission": cfg["name"],
        "seed": cfg["seed"],
        "agent": agent.name,
        "outcome": res.outcome,
        "found": agent.found,
        "inferred_location": loc,
        "location_confidence": confidence,
        "inferred_status": {"status": status, "probability": round(status_p, 3), "basis": status_note,
                            "distribution": {k: round(v, 3) for k, v in b.status_p.items()}},
        "evidence": {
            "initial_signal": agent.b["initial_signal"],
            "signal_readings": len(readings),
            "signal_dropouts": agent.stats["dropouts"],
            "strongest_reading": strongest,
            "suspicious_readings": agent.stats["suspicious"],
            "clues": clues,
            "cells_inspected_empty": inspected,
        },
        "recommended_next_step": nxt,
        "resources": {
            "steps": res.steps, "battery_spent": round(res.energy_spent, 1), "time_used": round(res.time_used, 1),
            "time_budget": cfg["resources"]["time"], "recharges": res.recharges,
            "stats": agent.stats,
        },
        "strategy_timeline": _segments(res.strategies),
        "evaluation_only": {
            "note": "Ground truth, used only to score the run. The agent never saw any of this.",
            "eve_position": truth["eve_pos"], "eve_status": truth["eve_status"],
            "best_guess_error_cells": round(res.map_error, 2),
            "decoy_clues": [c["id"] for c in truth["clues"] if c["decoy"]],
            "decoy_signal_sources": truth["decoys"],
        },
    }


def render_markdown(r: dict) -> str:
    loc, st, ev, rs = r["inferred_location"], r["inferred_status"], r["evidence"], r["resources"]
    where = (f"{loc['cell']} (confirmed, within {loc['within_cells']} cell)" if loc["confirmed"]
             else f"{loc['cell']} (best guess, within {loc['within_cells']} cells)")
    lines = [
        f"# Mission report: {r['mission']}",
        "",
        f"**Outcome:** {r['outcome']} ({'EVE found' if r['found'] else 'EVE not confirmed'})  ",
        f"**Agent:** {r['agent']}, seed {r['seed']}",
        "",
        "## Where is EVE?",
        f"- **Inferred location:** {where}",
        f"- **Confidence:** {r['location_confidence']:.0%}",
        f"- **Inferred status:** {st['status']} ({st['probability']:.0%}), {st['basis']}",
        "",
        "## Recommended next step",
        r["recommended_next_step"],
        "",
        "## Evidence",
        f"- Initial signal: strength {ev['initial_signal']['strength']:.3f}, bearing {ev['initial_signal']['bearing_deg']:.0f} deg",
        f"- Signal readings taken: {ev['signal_readings']} ({ev['signal_dropouts']} dropouts, {ev['suspicious_readings']} flagged as suspicious)",
    ]
    if ev["strongest_reading"]:
        s = ev["strongest_reading"]
        lines.append(f"- Strongest reading: {s['strength']:.2f} at {s['pos']} (step {s['step']})")
    if ev["clues"]:
        lines.append("- Clues found:")
        for c in ev["clues"]:
            lines.append(f"  - clue {c['id']} at {c['found_at']} points to {c['points_to']} (spread {c['spread']}): {c['assessment']}")
    else:
        lines.append("- Clues found: none")
    if ev["cells_inspected_empty"]:
        lines.append(f"- Inspected with nothing found: {ev['cells_inspected_empty']}")
    lines += [
        "",
        "## Resources",
        f"- Steps: {rs['steps']}, battery spent: {rs['battery_spent']}, time used: {rs['time_used']}/{rs['time_budget']}, recharges: {rs['recharges']}",
        "",
        "## How the strategy changed",
    ]
    for s in r["strategy_timeline"]:
        lines.append(f"- steps {s['from_step']}-{s['to_step']}: {s['phase']}")
    ev2 = r["evaluation_only"]
    lines += ["", "## Ground truth (evaluation only, never seen by the agent)",
              f"- EVE was at {ev2['eve_position']} ({ev2['eve_status']}); best guess was off by {ev2['best_guess_error_cells']} cells",
              f"- Decoy clues: {ev2['decoy_clues'] or 'none'}; decoy signal sources: {ev2['decoy_signal_sources'] or 'none'}", ""]
    return "\n".join(lines)


def write_all(res, cfg: dict, out: Path, gif: bool = False) -> dict:
    from . import viz

    out.mkdir(parents=True, exist_ok=True)
    res.trace.write_jsonl(out / "trace.jsonl")
    rep = build_report(res, cfg)
    (out / "report.json").write_text(json.dumps(rep, indent=2, default=_default))
    (out / "report.md").write_text(render_markdown(rep))
    np.savez_compressed(out / "belief.npz", belief=np.array(res.snapshots, dtype=np.float16))
    (out / "map.txt").write_text(viz.ascii_report(res))
    try:
        viz.save_png(res, out / "mission.png")
        if gif:
            viz.save_gif(res, out / "mission.gif")
    except ImportError:
        (out / "NO_IMAGES.txt").write_text("matplotlib is not installed; see map.txt for the ASCII view.\n")
    try:
        from . import html_viz

        html_viz.save_html(res, rep, out / "visualizer.html")
    except Exception:
        pass
    return rep
