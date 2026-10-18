# Mission report: mission_decoys

**Outcome:** found (EVE found)  
**Agent:** belief, seed 3

## Where is EVE?
- **Inferred location:** [16, 9] (confirmed, within 1 cell)
- **Confidence:** 100%
- **Inferred status:** damaged (100%), confirmed by direct inspection

## Recommended next step
EVE is damaged: bring a repair kit and stabilise her before moving her. She is 12 steps from a station; a round trip needs about 28 battery.

## Evidence
- Initial signal: strength 0.040, bearing -3 deg
- Signal readings taken: 12 (5 dropouts, 1 flagged as suspicious)
- Strongest reading: 0.32 at [2, 1] (step 5)
- Clues found: none
- Inspected with nothing found: [[16, 4], [18, 4], [18, 6], [17, 7], [17, 3], [17, 1], [14, 1], [14, 5], [14, 7]]

## Resources
- Steps: 113, battery spent: 131.0, time used: 121.0/160, recharges: 3

## How the strategy changed
- steps 1-8: localize
- steps 9-9: search-clues
- steps 10-22: localize
- steps 23-23: approach-eve
- steps 24-24: localize
- steps 25-41: approach-eve
- steps 42-42: search-clues
- steps 43-55: approach-eve
- steps 56-79: return
- steps 80-82: recharge
- steps 83-84: approach-eve
- steps 85-86: localize
- steps 87-87: approach-eve
- steps 88-88: localize
- steps 89-113: approach-eve

## Ground truth (evaluation only, never seen by the agent)
- EVE was at [17, 10] (damaged); best guess was off by 1.41 cells
- Decoy clues: [1, 2]; decoy signal sources: [[7, 10], [14, 1], [15, 0]]
