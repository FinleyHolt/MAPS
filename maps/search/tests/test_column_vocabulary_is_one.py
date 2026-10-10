"""The executor speaks ONE column vocabulary: the one it emits.

The columns were renamed from the ``d_<ref>`` / ``along_<ref>`` / ``vy_<ref>``
family to the phrase that names the relation (``distance to <ref>``,
``along <ref>``, ``ahead of <ref>``). The rename was applied where columns are
WRITTEN and missed where they are READ, and nothing failed:

  * ``_k.startswith("along_")`` and ``k.startswith("brg_")`` gated two
    score_by guidance notes. Neither had fired since the rename — including
    the one warning that ranking on a compass bearing is circular.
  * ``_grouped_keys`` collapsed per-instance families on
    ``^([a-z]+_)([a-z]+_)?``. ``distance to building_1`` has no underscore
    where that regex needs one, so every key became its own family and the
    400-name dump the function exists to prevent came straight back.
  * A dozen served strings still named columns the executor cannot emit, so
    a model following them writes a key that errors.

These are derived scans, so each carries a floor: a scan that finds nothing
passes, and a rename that deletes the vocabulary would make them all vacuous.
"""

from __future__ import annotations

import re
from pathlib import Path

from maps.search.analyze_scene.geom_program import (
    _COMPARE_COL_PREFIXES,
    _grouped_keys,
)

SRC = Path(__file__).resolve().parents[1] / "analyze_scene" / "geom_program.py"

#: Spellings the rename retired. A served string naming one of these tells the
#: model to use a column that does not exist; live code testing one of these as
#: a prefix is a branch that can never fire.
RETIRED = (
    "along_", "off_<", "d_<", "d_=", "dc_", "brg_gap_", "brg_",
    "vx_", "vy_", "left_/", "right_/",
)

#: Underscore-spelled prefixes that are NOT retired column names, each for a
#: stated reason. Add to this only with one.
#:   color_  — the surviving underscore column family: the palette word rides
#:             in the key (``color_navy``), so it is not a phrase.
#:   abs_    — a filter OPERATOR prefix (``abs_gt``), tested against the op
#:             string, never against a column key.
LIVE_UNDERSCORE_PREFIXES = ("color_", "abs_")


def _code_lines() -> list[tuple[int, str]]:
    return [(i, ln) for i, ln in enumerate(SRC.read_text().splitlines(), 1)
            if not ln.strip().startswith("#")]


def test_the_vocabulary_is_not_empty():
    """Floor for every scan below — all of them pass trivially if the
    prefix tuple is emptied or the module stops parsing."""
    assert len(_COMPARE_COL_PREFIXES) >= 8
    for required in ("distance to ", "along ", "ahead of ", "bearing from "):
        assert required in _COMPARE_COL_PREFIXES
    assert len(_code_lines()) > 3000


def test_no_served_string_names_a_retired_column():
    offenders = []
    for i, ln in _code_lines():
        for lit in re.findall(r'["\']([^"\']{4,})["\']', ln):
            if any(r in lit for r in RETIRED):
                offenders.append(f"{i}: {lit[:90]}")
                break
    assert not offenders, (
        "these strings name columns the executor cannot emit; a model "
        "following them writes a key that errors:\n  " + "\n  ".join(offenders))


def test_no_live_branch_tests_a_retired_prefix():
    """A ``startswith`` on a retired spelling is a branch that cannot fire."""
    offenders = []
    for i, ln in _code_lines():
        if "startswith" not in ln:
            continue
        for m in re.finditer(r'startswith\(\s*\(?([^)]*)\)?', ln):
            for s in re.findall(r'"([^"]*)"', m.group(1)):
                if s in LIVE_UNDERSCORE_PREFIXES or " " in s or not s:
                    continue
                if s.endswith("_") or s in RETIRED:
                    offenders.append(f"{i}: startswith({s!r}) -- {ln.strip()[:70]}")
    assert not offenders, (
        "dead prefix tests (the column is phrase-spelled now):\n  "
        + "\n  ".join(offenders))


def test_families_collapse_on_the_shipped_spelling():
    """The behaviour the scans above only approximate. A fan-out compare
    attaches one column per ref row; the summary must stay readable."""
    keys = [f"distance to building_{i}" for i in range(100)] + ["along road_5"]
    out = _grouped_keys(keys)
    assert "distance to building_* (100 instances" in out
    assert "along road_5" in out
    assert out.count("distance to building_") == 2  # the family + its example
    assert len(out) < 200, f"the 400-name dump is back: {out[:200]}"


def test_longest_prefix_wins_so_no_key_prints_twice():
    """'facing ' must not be swallowed by 'bearing from ' — the
    law the docstring states, and the one a startswith re-scan broke."""
    keys = ([f"facing b_{i}" for i in range(5)]
            + [f"bearing from b_{i}" for i in range(5)])
    out = _grouped_keys(keys)
    assert "facing b_* (5 instances" in out
    assert "bearing from b_* (5 instances" in out
    # Ten keys, two families, each named once.
    assert out.count("instances") == 2
