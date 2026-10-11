"""Served truth: every served note/annotation states the stored fact."""
from __future__ import annotations

import pytest

from maps.search.analyze_scene.geom_program import (
    compact_results,
    run_program,
)


def _prior(fid, x, y, name=None, tags=None):
    return {"fid": fid, "name": name, "xy": [float(x), float(y)],
            "centerline": None, "tags": tags or {}, "color": "",
            "search_text": " ".join(
                [name or ""] + [f"{str(k).replace('_', ' ')} "
                                f"{str(v).replace('_', ' ')}"
                                for k, v in (tags or {}).items()]).lower()}


def _grass_world():
    return [
        _prior("prior_1", 0, 0, tags={"landuse": "grass"}),
        _prior("prior_2", 10, 0, tags={"landuse": "grass"}),
        _prior("prior_3", 20, 0, tags={"landuse": "meadow"}),
        _prior("prior_4", 30, 0, tags={"building": "yes"}),
    ]


def _find(world, query):
    out = run_program([], world, {"ops": [
        {"ref": "r0", "op": "find",
         "args": {"provenance": "prior", "query": query}},
    ]})
    return out, compact_results(out["results"])["r0"]


# ---------------------------------------------------------------------------
# F1: the matched-note serves the truth about sub-threshold rows
# ---------------------------------------------------------------------------


def test_head_banner_endorses_only_a_name_tier_head():
    # A tag head is just the head; the number still rides the banner, labeled and
    # visibly weak.
    _, env = _find(_grass_world(), "landuse=grass")
    assert "Best: " not in env["hint"]
    assert "Head: " in env["hint"]
    assert "(match " in env["hint"]
    # ...while an exact NAME hit keeps the word it earned.
    world = _grass_world() + [_prior("prior_9", 40, 0, name="Mill Lane")]
    _, env = _find(world, "Mill Lane")
    assert "Best: " in env["hint"]
    assert "(match " in env["hint"]


# ---------------------------------------------------------------------------
# F2: ONE truth for [x,y] compare labels
# ---------------------------------------------------------------------------

def test_bare_xy_literal_attaches_the_xy_family():
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": [[5.0, 5.0]]}},
    ]})
    row = out["results"]["r1"][0]
    for col in ("distance to xy", "center distance to xy", "bearing from xy"):
        assert col in row, f"{col} missing: {sorted(row)}"


def test_labeled_xy_literal_is_the_named_columns_path():
    out = run_program([], _grass_world(), {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0",
                  "refs": [{"ref": [5.0, 5.0], "as": "gate"},
                           {"ref": [25.0, 5.0], "as": "corner"}]}},
    ]})
    row = out["results"]["r1"][0]
    assert "distance to gate" in row and "distance to corner" in row
    assert "distance to xy" not in row


# ---------------------------------------------------------------------------
# F6: palette-word zero-match affords the color path
# ---------------------------------------------------------------------------

def test_color_word_miss_names_the_color_column_path():
    """The COLUMN, not the op that attaches it."""
    _, env = _find(_grass_world(), "white")
    hint = env["hint"]
    assert env["len"] == 0
    assert "color_white" in hint
    assert "attaches when the color word is named" in hint
    assert "find{" not in hint and "score_by" not in hint


@pytest.mark.parametrize("query, column", [("zzqx OR white", "color_white"),
                                           ("silver", "color_gray")])
def test_a_colour_word_in_a_clause_or_a_synonym_names_the_canonical_column(query, column):
    _, env = _find(_grass_world(), query)
    assert env["len"] == 0
    assert column in env["hint"]


def test_non_color_miss_serves_no_color_affordance():
    _, env = _find(_grass_world(), "zzqx")
    assert env["len"] == 0
    assert "color_" not in env["hint"]
