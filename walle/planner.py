"""Decision helpers: expected information gain, inspection value, clue-search value.

Everything here works on the agent's own beliefs and known map.
"""
import math

import numpy as np

from .belief import Geometry, entropy
from .sensors import expected_strength, wrap_angle


def expected_info_gain(g: Geometry, p: np.ndarray, spec: dict, eps: float, cand: np.ndarray,
                       rng: np.random.Generator, K: int = 10) -> np.ndarray:
    """Monte-Carlo estimate of how many nats of uncertainty a signal scan at each
    candidate cell is expected to remove.

    Draw K plausible truths from the current belief, simulate the reading each would
    produce at the candidate, update, and average the posterior entropy. The same
    truths and noise are reused for every candidate (common random numbers), so the
    comparison between candidates is fair even though K is small."""
    H0 = entropy(p)
    ks = rng.choice(g.n, size=K, p=p / p.sum())
    noise = rng.standard_normal((K, 2))
    floor, sl = spec["floor"], spec["sigma_log"]
    sb = math.radians(spec["sigma_bearing_deg"])
    out = np.zeros(len(cand))
    for i, c in enumerate(cand):
        Ec = expected_strength(spec["S0"], spec["d0"], g.D[c])  # (n,) per hypothesised cell
        s_obs = np.maximum(Ec[ks], floor) * np.exp(sl * noise[:, 0])  # (K,)
        b_obs = g.A[c, ks] + sb * noise[:, 1]
        z = (np.log(np.maximum(s_obs, floor))[:, None] - np.log(np.maximum(Ec, floor))[None, :]) / sl
        dth = wrap_angle(b_obs[:, None] - g.A[c][None, :]) / sb
        dth[:, g.D[c] == 0] = 0.0
        L = (1 - eps) * np.exp(-0.5 * z * z - 0.5 * dth * dth) + eps
        post = p[None, :] * L
        post /= post.sum(axis=1, keepdims=True)
        Hpost = -(post * np.log(post + 1e-12)).sum(axis=1)
        out[i] = max(0.0, H0 - float(Hpost.mean()))
    return out


def clue_search_value(g: Geometry, p: np.ndarray, covered: np.ndarray, radius: float, detect_prob: float,
                      clues_left: float, clue_radius: float, clue_gain: float, cand: np.ndarray) -> np.ndarray:
    """Expected nats from an area scan at each candidate.

    The briefing says clues lie within `clue_radius` of EVE, so a cell is a likely clue
    site in proportion to belief mass near it. Cells already scanned are worth nothing."""
    if clues_left <= 0:
        return np.zeros(len(cand))
    reach = (g.D <= clue_radius).astype(float)
    w = reach @ p
    w = w / max(w.sum(), 1e-12)
    uncovered_w = w * (~covered.ravel())
    in_range = (g.D[cand] <= radius).astype(float)
    return clue_gain * clues_left * detect_prob * (in_range @ uncovered_w)


def inspect_value(mass3: np.ndarray, cand: np.ndarray, found_value: float, min_mass: float) -> np.ndarray:
    m = mass3[cand]
    return np.where(m >= min_mass, found_value * m, 0.0)
