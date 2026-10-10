"""The one ``[x, y]`` token grammar (``maps.util.xy_token``) every site
that accepts a world point reads."""
from __future__ import annotations

from maps.util.xy_token import parse_xy_token


# --------------------------------------------------------------------------- #
# the ONE token grammar                                                        #
# --------------------------------------------------------------------------- #


def test_xy_token_grammar():
    assert parse_xy_token("[1530, 1378]") == (1530.0, 1378.0)
    assert parse_xy_token("[1530.5,-378.25]") == (1530.5, -378.25)
    assert parse_xy_token(" 12, 34 ") == (12.0, 34.0)  # bare pair tolerated
    for bad in ("prior_12", "detected_5", "r3", "[1530]", "[a, b]",
                "[1, 2, 3]", "", None, 12):
        assert parse_xy_token(bad) is None, bad
