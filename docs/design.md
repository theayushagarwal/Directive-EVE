# Design notes

## The problem in one paragraph
EVE is somewhere on a grid the robot cannot see. The robot gets one noisy first reading of her signal, then
has to spend a limited battery and limited time on moves, scans, clue searches and a final inspection, while
always keeping enough battery to get back to a charging station. It must learn clue locations only by
looking. The agent therefore has to answer three questions at every step: *where is she probably, what would
tell me the most, and can I afford to find out?*

## Belief: a probability grid, updated by Bayes' rule
`walle/belief.py` keeps `p[cell]`, the probability EVE is in each cell, plus a three-way distribution over her
status (active, damaged, dormant).

| Evidence | How it updates the belief |
|---|---|
| Signal reading (strength + bearing) | Multiply by how likely that reading is from each cell: log-normal in strength around the falloff curve, Gaussian in bearing angle. |
| Spoofed signals | The likelihood is a mixture, `(1 - eps) * model + eps * flat` (eps = 0.15). One bad reading cannot zero out the truth, and a reading the belief finds very unlikely is counted as *suspicious*. |
| Signal dropout | Evidence about her **condition**, not her position: damaged and dormant units drop out far more often, so dropouts shift the status distribution only. |
| Trail clue | A mixture too: a clue is treated as 70% reliable (it may be a decoy), so one clue pulls belief toward its target without ruling anything out. |
| Failed inspection | Rules out the 3x3 block around the robot. |
| Wall discovered | That cell holds no probability. |

## Decisions: information per unit of cost
`BeliefAgent.choose` scores every reachable cell for three kinds of action and picks the best ratio of
**value / (battery cost + time cost)**:

- **Scan for signal.** Value is the *expected information gain*: draw plausible truths from the current
  belief, simulate the reading each would produce at that cell, update, and measure how much entropy drops.
  The same random draws are reused for every candidate so the comparison is fair even with few samples.
- **Inspect.** Value is the belief mass within one cell, times the worth of actually finding EVE.
- **Search for clues.** Value is the expected number of undiscovered clues in unscanned ground, weighted by
  where belief says clues are likely (the briefing says they lie within a known radius of EVE).

Two practical details: the agent **commits to a plan** unless a new option beats it by 30%, so it does not
dither between near-equal targets; and repeat scans from the same cell are **discounted**, because readings
from one spot share systematic error and a new vantage point triangulates better.

## Resources: always able to get home
Turn-back decisions use only cells *already seen to be free*. That guarantees the route exists and is as long
as the agent thinks it is. (An earlier version planned the way home through unexplored cells; each newly
discovered wall added a detour, and a robot ran out of battery three steps from the station. The safety
margin must also exceed the costliest action, which the config loader enforces.) On top of that, a final
**guard** applies to every agent, including the baselines and an LLM: any action that would leave too little
battery to get home is replaced by a return move, a recharge, or finishing.

## What the agent is allowed to know
`env.py` is the only module that touches ground truth. The agent receives a *briefing* (map size, start,
station positions, resource budget, sensor specification, first reading, how many clues to expect) and
observations. A test parses the agent modules' imports and fails if any of them import `world` or `env`.

## Honest limits of the model
- The signal model ignores walls (no attenuation or reflection), so it is simpler than real radio.
- The agent knows the sensor specification exactly. A real robot would have to calibrate it.
- The clue gain, found-value and repeat discount are hand-set constants, not learned.
- Status inference only uses dropout counts, and a spoofed signal never drops out, so spoofing nudges the
  status estimate toward "active".
