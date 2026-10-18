# Mission report: mission_easy

**Outcome:** found (EVE found)  
**Agent:** belief, seed 7

## Where is EVE?
- **Inferred location:** [4, 6] (confirmed, within 1 cell)
- **Confidence:** 100%
- **Inferred status:** active (100%), confirmed by direct inspection

## Recommended next step
EVE is responsive: open a comms link and escort her to the nearest station. She is 8 steps from a station; a round trip needs about 20 battery.

## Evidence
- Initial signal: strength 0.446, bearing 69 deg
- Signal readings taken: 2 (0 dropouts, 1 flagged as suspicious)
- Strongest reading: 0.68 at [1, 5] (step 7)
- Clues found: none
- Inspected with nothing found: [[1, 5], [3, 5]]

## Resources
- Steps: 13, battery spent: 15.0, time used: 13.0/160, recharges: 0

## How the strategy changed
- steps 1-1: localize
- steps 2-6: approach-eve
- steps 7-7: localize
- steps 8-13: approach-eve

## Ground truth (evaluation only, never seen by the agent)
- EVE was at [3, 7] (active); best guess was off by 1.41 cells
- Decoy clues: [0]; decoy signal sources: none
