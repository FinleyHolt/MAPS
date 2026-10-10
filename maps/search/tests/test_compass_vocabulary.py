"""ONE compass table, and every op declares which subset of it it hears.

There were five hand-written direction-word lists in a tree whose own
``geom_compute`` docstring says "a second hand-written copy is how a silent
mirror starts": ``COMPASS_UNIT`` (4), ``cardinal_side.CARDINALS`` (4),
``python_surface._FURTHER`` (4) and
``python_surface._DIAGONALS`` (4) -- while ``_compass8`` emitted EIGHT words
for the ``facing <ref>`` column. Eight out, four in: a model that read
``facing X == "southeast"`` had no operation that would take the word back,
and ``stand(facing="southeast")`` fell silently through to a lookup for a
place named "southeast".

The rule this pins: ``COMPASS_UNIT`` is the table, it carries eight, and an
op that hears fewer says WHICH fewer by naming ``CARDINALS``. Widening the
table can then never silently widen an operation, and narrowing an op can
never silently invent a word the geometry does not have.

VACUITY FLOOR. Every assertion below would pass against empty collections,
which is exactly how a derived guard rots into a clean confident zero. The
floors come first: the table holds eight, the subset holds four, and both
are non-empty before anything is compared.
"""
from __future__ import annotations

import math

import pytest

from maps.search.analyze_scene import cardinal_side
from maps.search.analyze_scene.geom_compute import (
    CARDINALS,
    COMPASS_UNIT,
    _COMPASS8,
    _compass8,
    cardinal,
)

_EIGHT = {"north", "northeast", "east", "southeast",
          "south", "southwest", "west", "northwest"}


def test_the_table_is_eight_winds_and_the_subset_is_four():
    """The vacuity floor for every other test in this file."""
    assert len(COMPASS_UNIT) == 8, sorted(COMPASS_UNIT)
    assert set(COMPASS_UNIT) == _EIGHT, sorted(COMPASS_UNIT)
    assert len(CARDINALS) == 4, sorted(CARDINALS)
    assert CARDINALS < set(COMPASS_UNIT), "the subset must be PROPER"


def test_every_unit_vector_round_trips_through_its_own_word():
    """word -> unit vector -> word. A mirrored or transposed entry breaks
    here rather than three layers downstream in a served column."""
    for word, u in COMPASS_UNIT.items():
        assert math.isclose(math.hypot(*u), 1.0, rel_tol=1e-12), (word, u)
        assert cardinal([0.0, 0.0], list(u)) == word, (word, u)


def test_the_frame_convention_is_stated_once_and_holds():
    # x = east, y = NORTH. Every directional op in the tree rides this.
    assert COMPASS_UNIT["north"] == (0.0, 1.0)
    assert COMPASS_UNIT["east"] == (1.0, 0.0)
    # ...and the diagonals sit exactly between their two cardinals, which is
    # what makes "southeast" a projectable axis rather than a label.
    for diag, (a, b) in (("northeast", ("north", "east")),
                         ("southeast", ("south", "east")),
                         ("southwest", ("south", "west")),
                         ("northwest", ("north", "west"))):
        want = tuple((COMPASS_UNIT[a][i] + COMPASS_UNIT[b][i]) / math.sqrt(2)
                     for i in (0, 1))
        assert all(math.isclose(g, w, abs_tol=1e-12)
                   for g, w in zip(COMPASS_UNIT[diag], want)), diag


def test_compass8_words_are_exactly_the_table_keys():
    """The wedge table and the unit table name the same eight things."""
    assert {name for _hi, name in _COMPASS8} == set(COMPASS_UNIT)
    # and the binning agrees with the vectors at every wedge centre
    for i, word in enumerate(("north", "northeast", "east", "southeast",
                              "south", "southwest", "west", "northwest")):
        assert _compass8(i * 45.0) == word, (i, word)


# --- the declared subsets ---------------------------------------------------
# One test per op that hears fewer than eight, each stating its reason. A new
# four-wind op with no entry here is the thing this file exists to catch.


def test_the_band_rule_declares_the_four_cardinals():
    """A diagonal has no single orthogonal axis to band on, so cardinal_side
    offers four -- DERIVED from CARDINALS, never a fifth copy."""
    assert set(cardinal_side.CARDINALS) == set(CARDINALS)
    for w in CARDINALS:
        assert cardinal_side.CARDINALS[w] == COMPASS_UNIT[w], w
    # and it refuses a diagonal loudly rather than answering badly
    road = [[0.0, y] for y in range(-100, 101)]
    with pytest.raises(ValueError):
        cardinal_side.cardinal_offset(road, (50.0, 0.0), "northeast")


def test_no_op_hears_a_word_the_geometry_cannot_compute():
    """The direction of the guard that actually matters: an op may narrow the
    table, never widen it. A word in an op and not in COMPASS_UNIT would be a
    spelling with no vector behind it."""
    for name, words in (("cardinal_side.CARDINALS", set(cardinal_side.CARDINALS)),
                        ("geom_compute.CARDINALS", set(CARDINALS))):
        assert words, f"{name} is empty -- vacuity"
        assert words <= set(COMPASS_UNIT), (name, sorted(words - set(COMPASS_UNIT)))
