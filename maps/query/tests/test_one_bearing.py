"""One bearing: the benches' signed compass bearing and the library's 0-360 one are the same float."""

from __future__ import annotations

import math
import random

from maps.query import spatial_math as G


def test_the_signed_bearing_is_bit_identical_to_the_benches_formulas_and_wraps_to_the_librarys():
    rng = random.Random(0)
    for _ in range(20000):
        a = (rng.uniform(-1e4, 1e4), rng.uniform(-1e4, 1e4))
        b = (rng.uniform(-1e4, 1e4), rng.uniform(-1e4, 1e4))
        if rng.random() < 0.1:
            b = (a[0] + rng.choice((0.0, 1.0, -1.0)), a[1] + rng.choice((0.0, 1.0, -1.0)))
        # The formulas citynav's action_snap and flight's measures each spelled out.
        signed = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
        assert G.bearing_deg(a, b) == signed
        assert G.bearing(a, b) == signed % 360.0
