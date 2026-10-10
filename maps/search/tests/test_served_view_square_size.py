"""The served prompt states ONE view-square size, and the sensor rig owns it.

The prompt said the view square had "side 2x(altitude - ground) m" and, ~40
lines below in the same string, that the rig's footprint runs "~321 x 241 m".
At the 914.4 m AGL the calcity config flies, those are 3,344,509 m2 and
77,278 m2 -- 43x apart, in one prompt, and the rig was the right one.

The fix is a derivation and not a better constant: the prompt now says the
square IS the rig's footprint at the flight's altitude and states no size of
its own, so there is one producer and it is the sensor spec. That leaves a
derived-absence test, which passes when the thing it looks for is deleted --
so the floor below asserts the prompt still describes the view square at
all; benchmarks/calcity/tests/test_served_rig_footprint.py checks the rig's
own prose against ``footprint_m`` at the height the config flies.
"""
from __future__ import annotations

import re

from maps.perception.geometry.sensor_spec import resolve_sensor, served_description
from maps.search.parse_compile import reasoner_agent as ra


def _served() -> str:
    """The Oryx rig's served text at 3,000 ft AGL (the Cal City config's
    height; benchmarks/calcity/tests checks the config's own)."""
    return ra.served_system_text(
        sensor_description=served_description(resolve_sensor("oryx_orx32_25mm_nadir"), 914.4))


def test_the_prompt_states_no_view_square_size_of_its_own():
    text = _served()
    # VACUITY FLOOR: the paragraph this guards must still be there.
    assert "nadir view square" in text
    assert "nothing outside it is seen" in text
    # ...and it derives rather than restating an arithmetic that was wrong
    # by 43x on the flown config.
    assert "altitude - ground" not in text
    assert "2x(altitude" not in text


def test_a_run_with_no_rig_states_no_footprint_at_all():
    """An unconfigured rig is an honest silence, not a formula. Nothing
    downstream reads a size out of the prompt -- ``obs_span_m``, which
    bounds the look crop, is computed from the sensor and never parsed
    from this text."""
    text = ra.served_system_text()
    assert "nadir view square" in text
    assert not re.search(r"~\d+ x \d+ m", text), text
