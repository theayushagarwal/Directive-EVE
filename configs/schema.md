# Mission config format

Missions are JSON or YAML. **Every field is optional**: anything omitted takes the default shown, and anything
under `hidden` that is omitted is generated from `seed`, so the same file always gives the same mission.
`mission_easy.yaml` and `mission_decoys.yaml` spell everything out explicitly; `scripts/make_configs.py` regenerates them.

Coordinates are `[x, y]`, with `x` to the right and `y` **downward**. Bearings are degrees, 0 = east, 90 = south.

## Visible to the robot (the "briefing")

| Field | Default | Meaning |
|---|---|---|
| `name`, `seed` | `mission`, `0` | Label, and the seed for everything random. |
| `map.width`, `map.height` | 21, 21 | Grid size (7 to 60). |
| `map.wall_density` | 0.12 | Fraction of cells that are walls when `walls` is not given. |
| `map.walls` | generated | Explicit list of wall cells `[[x, y], ...]`. |
| `start` | `[1, 1]` | Where the robot begins. |
| `stations` | start + one generated | Charging stations the robot knows about. |
| `resources.battery` | 100 | Battery capacity; the robot starts full. |
| `resources.time` | 160 | Time budget. The mission ends at zero. |
| `resources.recharge_amount` | 30 | Battery gained per `RECHARGE` (only at a station). |
| `resources.costs` | move 1, scan_signal 2, scan_area 3, inspect 1 | Battery per action. Recharging costs no battery. |
| `resources.time_costs` | move 1, scan_signal 1, scan_area 2, inspect 1, recharge 3 | Time per action. |
| `sensors.signal` | S0 1.0, d0 5.0, sigma_log 0.25, sigma_bearing_deg 18, floor 0.005 | Signal model: strength is `S0 / (1 + (d/d0)^2)` with log-normal noise; bearing has Gaussian noise. |
| `sensors.area` | radius 3, detect_prob 0.9 | Clue search radius and per-clue detection chance. |
| `dropout` | active 0.05, damaged 0.35, dormant 0.75 | Chance a signal scan returns nothing, by EVE's status. |
| `initial_signal` | generated | First reading `{bearing_deg, strength, timestamp}`, taken at `start`. |
| `agent.safety_margin` | 4 | Battery kept in reserve for the trip home. Must exceed the costliest action by at least 1. |
| `agent.spoof_epsilon` | 0.15 | How much the robot allows that any signal reading is spoofed. |
| `agent.clue_reliability` | 0.7 | How far the robot trusts a trail clue. |

## Hidden from the robot (the simulator's ground truth)

| Field | Default | Meaning |
|---|---|---|
| `hidden.eve` | generated | `{pos: [x, y], status: active/damaged/dormant}`. |
| `hidden.clues` | generated | List of `{pos, target, sigma, decoy}`. A clue found at `pos` points to `target` (spread `sigma`). A `decoy: true` clue points somewhere wrong. |
| `hidden.n_clues` | 4 | How many clues to generate (also told to the robot as a hint). |
| `hidden.decoy_clue_fraction` | 0.25 | Share of generated clues that are decoys. |
| `hidden.clue_sigma` | 2.0 | Spread of generated clues. |
| `hidden.clue_radius` | 9 | Generated clues lie within this distance of EVE (told to the robot as a hint). |
| `hidden.decoys` | generated | Cells emitting EVE-like signals `[[x, y], ...]`. |
| `hidden.n_decoys` | 0 | How many decoy signal sources to generate. |
| `hidden.spoof_prob` | 0.0 | Chance that a signal scan hears a decoy instead of EVE. |
| `hidden.eve_dist_range` | `[6, ~0.35 x battery]` | Allowed walking distance from the nearest station when placing EVE. |

## Minimal example

```yaml
name: tiny
seed: 11
resources: {battery: 80, time: 120}
hidden: {n_decoys: 2, spoof_prob: 0.25}
```

## Actions and what they return

| Action | Effect | Observation |
|---|---|---|
| `MOVE` (N/E/S/W) | Move one cell; walls and the map edge block (still costs). | `moved` or `bumped` |
| `SCAN_SIGNAL` | Noisy strength + bearing of the signal, or nothing (dropout). | `reading` or `null` |
| `SCAN_AREA` | Reveals clues within the radius (each found with `detect_prob`). | list of clues |
| `INSPECT` | Finds EVE if within 1 cell (ends the mission), reveals her status. | `found`, `status` |
| `RECHARGE` | At a station only. | new battery |
| `FINISH` | The agent ends the mission. | |

Every observation also reports position, battery, time left, whether the robot is at a station, and what is in
the four neighbouring cells (free / wall / edge), which is how the robot learns the map.

A mission ends as `found`, `timeout`, `stranded` (battery empty away from a station), or `abandoned`.
