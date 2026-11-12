"""ONE compass table, and every op declares which subset of it it hears."""

from __future__ import annotations

import math

import pytest

from maps.query.spatial_math import (
    CARDINALS,
    COMPASS_UNIT,
    _COMPASS8,
    _compass8,
    cardinal,
    cardinal_offset,
)

_EIGHT = {"north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"}


def test_the_table_is_eight_winds_and_the_subset_is_four():
    """The vacuity floor for every other test in this file."""
    assert len(COMPASS_UNIT) == 8, sorted(COMPASS_UNIT)
    assert set(COMPASS_UNIT) == _EIGHT, sorted(COMPASS_UNIT)
    assert len(CARDINALS) == 4, sorted(CARDINALS)
    assert CARDINALS < set(COMPASS_UNIT), "the subset must be PROPER"


def test_every_unit_vector_round_trips_through_its_own_word():
    """word -> unit vector -> word."""
    for word, u in COMPASS_UNIT.items():
        assert math.isclose(math.hypot(*u), 1.0, rel_tol=1e-12), (word, u)
        assert cardinal([0.0, 0.0], list(u)) == word, (word, u)


def test_the_frame_convention_is_stated_once_and_holds():
    # x = east, y = NORTH. Every directional op in the tree rides this.
    assert COMPASS_UNIT["north"] == (0.0, 1.0)
    assert COMPASS_UNIT["east"] == (1.0, 0.0)
    # ...and the diagonals sit exactly between their two cardinals, which is
    # what makes "southeast" a projectable axis rather than a label.
    for diag, (a, b) in (
        ("northeast", ("north", "east")),
        ("southeast", ("south", "east")),
        ("southwest", ("south", "west")),
        ("northwest", ("north", "west")),
    ):
        want = tuple((COMPASS_UNIT[a][i] + COMPASS_UNIT[b][i]) / math.sqrt(2) for i in (0, 1))
        assert all(math.isclose(g, w, abs_tol=1e-12) for g, w in zip(COMPASS_UNIT[diag], want)), (
            diag
        )


def test_compass8_words_are_exactly_the_table_keys():
    """The wedge table and the unit table name the same eight things."""
    assert {name for _hi, name in _COMPASS8} == set(COMPASS_UNIT)
    # and the binning agrees with the vectors at every wedge centre
    for i, word in enumerate(
        ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    ):
        assert _compass8(i * 45.0) == word, (i, word)


# --- the declared subsets ---------------------------------------------------
# One test per op that hears fewer than eight, each stating its reason. A new
# four-wind op with no entry here is the thing this file exists to catch.


def test_the_band_rule_declares_the_four_cardinals():
    """A diagonal has no single orthogonal axis to band on, so the band rule
    refuses one loudly rather than answering badly."""
    road = [[0.0, y] for y in range(-100, 101)]
    with pytest.raises(ValueError):
        cardinal_offset(road, (50.0, 0.0), "northeast")
