import math

import numpy as np

from walle.belief import Belief
from walle.sensors import bearing_rad, expected_strength

SPEC = {"S0": 1.0, "d0": 5.0, "sigma_log": 0.25, "sigma_bearing_deg": 18.0, "floor": 0.005}
DROP = {"active": 0.05, "damaged": 0.35, "dormant": 0.75}
W = H = 15
EVE = (10, 9)


def fresh():
    return Belief(W, H, SPEC, DROP)


def reading(at, src=EVE):
    d = math.dist(at, src)
    return float(expected_strength(1.0, 5.0, d)), bearing_rad(at, src)


def test_belief_is_a_probability_distribution_after_every_kind_of_update():
    b = fresh()
    assert abs(b.p.sum() - 1) < 1e-9
    b.apply_signal((2, 2), *reading((2, 2)))
    b.apply_dropout(True)
    b.apply_clue({"target": [10, 9], "sigma": 2.0})
    b.apply_negative_inspect((5, 5))
    b.mark_wall((7, 7))
    assert abs(b.p.sum() - 1) < 1e-9 and (b.p >= 0).all()


def test_clean_readings_from_several_places_pin_down_eve():
    b = fresh()
    for pos in [(2, 2), (2, 12), (12, 2), (8, 8)]:
        b.apply_signal(pos, *reading(pos))
    assert math.dist(b.map_cell(), EVE) <= 1.5
    assert b.mass_near(EVE, 2) > 0.85           # the mass sits on EVE...
    assert b.entropy() < 0.6 * math.log(W * H)  # ...and uncertainty fell well below a uniform prior


def test_a_single_spoofed_reading_cannot_erase_the_truth():
    b = fresh()
    for pos in [(2, 2), (2, 12), (12, 2)]:
        b.apply_signal(pos, *reading(pos))
    truth_cell_before = b.p[EVE[1] * W + EVE[0]]
    b.apply_signal((8, 8), *reading((8, 8), src=(1, 13)))  # strong reading pointing the wrong way
    assert b.p[EVE[1] * W + EVE[0]] > 0.1 * truth_cell_before


def test_a_spoofed_reading_is_flagged_as_surprising():
    b = fresh()
    for pos in [(2, 2), (2, 12), (12, 2), (8, 8)]:
        b.apply_signal(pos, *reading(pos))
    good = b.apply_signal((5, 5), *reading((5, 5)))
    bad = b.apply_signal((5, 5), *reading((5, 5), src=(1, 13)))
    assert bad < good


def test_dropouts_shift_the_status_estimate_toward_damaged_or_dormant():
    b = fresh()
    for _ in range(6):
        b.apply_dropout(True)
    assert b.status_p["active"] < 0.05
    c = fresh()
    for _ in range(6):
        c.apply_dropout(False)
    assert c.status_estimate()[0] == "active" and c.status_p["dormant"] < 0.01


def test_negative_inspection_rules_out_the_block():
    b = fresh()
    b.apply_negative_inspect((5, 5))
    assert b.mass_near((5, 5), 1) == 0.0 and abs(b.p.sum() - 1) < 1e-9


def test_walls_hold_no_probability():
    b = fresh()
    b.mark_wall((3, 3))
    assert b.p[3 * W + 3] == 0.0


def test_contradictory_evidence_does_not_crash_or_produce_nans():
    b = fresh()
    b.p[:] = 0
    b.p[0] = 1.0
    b.apply_negative_inspect((0, 0))  # removes the only cell with mass
    assert np.isfinite(b.p).all() and abs(b.p.sum() - 1) < 1e-9


def test_clue_pulls_belief_toward_its_target_but_not_absolutely():
    b = fresh()
    b.apply_clue({"target": [10, 9], "sigma": 1.5})
    near = b.mass_near((10, 9), 2)
    assert near > 0.05 and b.p.min() > 0  # mixture: no cell is ruled out by one clue
