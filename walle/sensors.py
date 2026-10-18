"""The signal model, shared by the simulator (to generate readings) and the agent
(to interpret them). It is the robot's published sensor spec, not hidden truth."""
import math

import numpy as np


def expected_strength(S0, d0, dist):
    """Inverse-square-ish falloff; distances below half a cell are clamped."""
    return S0 / (1.0 + (np.maximum(dist, 0.5) / d0) ** 2)


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def bearing_rad(src, dst) -> float:
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def sample_signal(rng, spec: dict, src, source) -> dict:
    """A noisy reading taken at `src` of a source at `source`."""
    d = math.hypot(source[0] - src[0], source[1] - src[1])
    true_s = max(float(expected_strength(spec["S0"], spec["d0"], d)), spec["floor"])
    strength = true_s * math.exp(spec["sigma_log"] * rng.standard_normal())
    bearing = bearing_rad(src, source) + math.radians(spec["sigma_bearing_deg"]) * rng.standard_normal()
    return {"strength": float(strength), "bearing_deg": float(math.degrees(wrap_angle(bearing)))}
