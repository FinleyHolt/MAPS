"""The served prompt states ONE view-square size, and the sensor rig owns it."""

from __future__ import annotations

import re

from maps.perception.geometry.sensor_spec import resolve_sensor
from maps.perception.ortho.sensor_frame import served_description
from maps.mission.served_text import served_system_prompt


def _served() -> str:
    return served_system_prompt(
        served_description(resolve_sensor("oryx_orx32_25mm_nadir"), agl_m=914.4)
    )


def test_the_prompt_states_no_view_square_size_of_its_own():
    text = _served()
    # VACUITY FLOOR: the paragraph this guards must still be there.
    assert "nadir view square" in text
    assert "nothing outside it is seen" in text
    # ...and it carries no formula for the size.
    assert "altitude - ground" not in text
    assert "2x(altitude" not in text


def test_a_run_with_no_rig_states_no_footprint_at_all():
    """An unconfigured rig is an honest silence, not a formula."""
    text = served_system_prompt()
    assert "nadir view square" in text
    assert not re.search(r"~\d+ x \d+ m", text), text
