# WALL-E and EVE's Signal

An autonomous agent that explores a simulated environment, gathers clues, manages a limited battery and
time budget, and works out where EVE is and what state she is in, then reports its conclusion and a
recommended next move.

The agent keeps a **probability grid** over EVE's location, updates it by Bayes' rule from every noisy reading
and clue, and at each step picks the action with the best **expected information per unit of battery and time**,
while always keeping enough battery to get home. Every action, observation, decision and reason is logged.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1   | Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# One-click demo (runs missions, generates all deliverables, opens web UI):
python scripts/demo.py

# Or run individual commands:
python scripts/run_mission.py configs/mission_easy.yaml   --out runs/mission_easy   --gif
python scripts/run_mission.py configs/mission_decoys.yaml --out runs/mission_decoys --gif
python scripts/visualize.py runs/mission_decoys                # interactive web UI in browser
python -m pytest -q                                        # 63 tests, about 1 minute
python scripts/benchmark.py --seeds 20 --out runs/benchmark   # about 4 minutes
```

Options: `--agent belief|gradient|random|llm`, `--seed N` to override the config seed, `--gif`, `--quiet`.
Everything is plain Python and runs on Windows, Mac and Linux.

## What a run produces (`--out DIR`)

| File | Contents |
|---|---|
| `visualizer.html` | **Interactive Web UI**: step scrubber, real-time probability heatmap, path playback, telemetry gauges, and natural language reasoning inspector. |
| `trace.jsonl` | One line per step: action, observation, the agent's strategy, its **reason in words**, the top alternatives it weighed, and a belief summary. |
| `report.md` / `report.json` | Evidence found, inferred location and status, confidence, recommended next step, resources used, how the strategy changed, plus a clearly separated ground-truth block for scoring. |
| `mission.png` | Path coloured by strategy, clues, stations, final belief, and an uncertainty/battery timeline. |
| `mission.gif` | Animated belief and path (with `--gif`). |
| `map.txt` | ASCII version of the map and belief. |
| `belief.npz` | The belief grid at every step. |

Saved examples are in `runs/mission_easy/` and `runs/mission_decoys/`.

## Environment design

Full field-by-field reference: [`configs/schema.md`](configs/schema.md). Summary:

- **World.** A grid with walls, one or more charging stations, EVE (hidden position, hidden status: active,
  damaged or dormant), and trail clues. Anything not pinned in the config is generated from the seed, so a
  mission is reproducible from `(config, seed)`.
- **The signal.** Strength falls off with distance, `S0 / (1 + (d/d0)^2)`, with log-normal noise. A reading
  also gives a noisy bearing. A damaged or dormant EVE sometimes returns *nothing*. The first reading (bearing,
  strength, timestamp) comes with the briefing.
- **Clues.** Found only by an area scan. A trail clue points to where EVE probably is, with a spread. Some are
  decoys that point somewhere wrong.
- **Decoys and noise (the advanced options).** Decoy transmitters can answer a signal scan instead of EVE
  (`spoof_prob`), and the noise levels are configurable.
- **Actions and costs.** Move, scan signal, scan area, inspect, recharge (stations only), finish. Each costs
  battery and time; defaults are in the schema.
- **What the agent sees.** A briefing and observations only. EVE's position, her status and the clue locations
  never leave `env.py` except through sensor readings, and a test checks that the agent modules do not
  import the world or the environment.
- **How a mission ends.** `found` (an inspection within 1 cell of EVE), `timeout`, `stranded` (battery empty
  away from a station), or `abandoned` (the agent decides to stop).

## How the agent decides

Short version; the full reasoning is in [`docs/design.md`](docs/design.md).

1. **Belief.** Each cell's probability is multiplied by how likely the latest reading would be if EVE were
   there. Signal readings and clues use *mixtures* so a single spoofed reading or decoy clue cannot erase the
   truth. Signal dropouts update the estimate of her *condition* instead of her position.
2. **Choice.** It scores scanning, inspecting and clue searching at every reachable cell by value divided by
   cost, where scanning is valued by the entropy it is expected to remove. Plans adapt every step as the belief changes.
3. **Turning back and recharging.** It tops up at stations, and turns back while the route home, over cells
   already seen to be free, is still affordable with a safety margin.
4. **Reporting.** Confidence is the belief mass within 2 cells of the best guess. If confidence is low the
   report recommends *where to take more readings* rather than guessing.

## Results

240 missions: 20 random missions (seeds 1000 to 1019) in each of four scenarios, played by three agents.
`runs/benchmark/` has the raw per-mission CSV. Intervals are 95% Wilson intervals, because 20 runs gives a
wide range.

| scenario | agent | found | 95% interval | stranded | mean steps (found) | mean battery used |
|---|---|---|---|---|---|---|
| clean | belief | 19/20 | 76% to 99% | 0 | 52 | 69 |
| clean | gradient | 16/20 | 58% to 92% | 0 | 53 | 96 |
| clean | random | 1/20 | 1% to 24% | 0 | 73 | 168 |
| noisy | belief | 19/20 | 76% to 99% | 0 | 69 | 92 |
| noisy | gradient | 11/20 | 34% to 74% | 0 | 53 | 127 |
| noisy | random | 1/20 | 1% to 24% | 0 | 73 | 168 |
| decoys | belief | 18/20 | 70% to 97% | 0 | 69 | 95 |
| decoys | gradient | 9/20 | 26% to 66% | 0 | 69 | 145 |
| decoys | random | 1/20 | 1% to 24% | 0 | 73 | 168 |
| large | belief | 18/20 | 70% to 97% | 0 | 66 | 102 |
| large | gradient | 13/20 | 43% to 82% | 0 | 61 | 161 |
| large | random | 2/20 | 3% to 30% | 0 | 34 | 253 |

What this does and does not show:

- With **clean** sensors a simple reactive agent does nearly as well (the intervals overlap), so the extra
  machinery is not needed there.
- The belief agent's advantage appears with **noise, decoys and bigger maps**, where the reactive agent
  trusts every reading and finds EVE in only about half the missions (11/20 noisy, 9/20 decoys, 13/20 large,
  against 18 or 19 of 20 for the belief agent). It also uses noticeably less battery.
- 20 missions per cell cannot separate small differences, and all scenarios are generated by the same
  simulator the agent's sensor model was designed around.
- No agent was ever stranded, because every agent shares the battery guard. The comparison is about
  *finding EVE*, not about safety.
- Tuning was done on other seeds (100 to 111); the benchmark seeds were not used for tuning.

## The two saved missions

- `mission_easy`: clean signal, finds EVE in 13 steps without recharging.
- `mission_decoys`: three decoy transmitters, 30% spoofing and a damaged EVE. The agent is misled into nine
  empty inspections, runs low, turns back and recharges (steps 56 to 82), then re-plans and finds her in 113
  steps. `runs/mission_decoys/mission.png` shows the strategy changes, and `report.md` lists the strategy timeline.

## Optional: LLM versus classical planner

`--agent llm` lets a language model choose each action from a text summary of the state and belief, via the
Groq API (`GROQ_API_KEY`). The battery guard still applies and unusable answers fall back to a scan. **I have
not run this against a live model**; the tests use a stub, so no LLM result is reported here.

## Tests

`python -m pytest -q` runs 63 tests covering: config validation, exact cost accounting and mission-end
rules, no hidden truth in observations, belief normalisation and robustness to spoofed readings, the agent
never importing the world, no agent ever stranded or overdrawn across scenarios, turning back and
recharging, reproducibility, and the trace, report and image outputs.

## Limitations

- The signal model ignores walls, and the agent is told the sensor specification exactly.
- Value constants (found-value, clue gain, repeat-scan discount) are hand-set, not learned.
- Status inference uses dropout counts only, and spoofed signals never drop out, so spoofing nudges the estimate
  toward "active".
- The agent can finish with the battery very low (it turns back as late as it safely can), so there is no
  slack for sensor or model errors.
- A failed final inspection can be a *near miss*: inspecting only finds EVE within 1 cell.
- Single agent only; multi-agent coordination and physics simulation are not implemented.
