"""The grid header serves column keys verbatim, in the spelling the executor accepts."""
from __future__ import annotations

import re

from maps.search.analyze_scene.geom_program import KEY_PATTERN, _render_grid_rows


def test_header_renders_keys_verbatim_with_units_separate():
    cols = ["distance to new_park_street", "facing new_park_street",
            "footprint_area_m2"]
    header = _render_grid_rows([], cols).splitlines()[0]
    for c in cols:
        assert c in header, header
    assert "(m)" not in header, header
    assert "(deg)" not in header, header
    for c in ("distance to new_park_street", "facing new_park_street"):
        assert re.match(KEY_PATTERN, c), (KEY_PATTERN, c)
        assert not re.match(KEY_PATTERN, c + "(m)")
