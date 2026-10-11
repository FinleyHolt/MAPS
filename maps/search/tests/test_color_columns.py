#!/usr/bin/env python
"""The lazy color-column family ``color_<word>``."""
from __future__ import annotations

from maps.search.analyze_scene.geom_program import run_program, ProgramError
from maps.search.analyze_scene import geom_compute as G

_CANDS = [
    {"fid": "d0", "xy": [0.0, 0.0], "class": "car", "color": "blue", "color_conf": 0.9},
    {"fid": "d1", "xy": [40.0, 0.0], "class": "car", "color": "purple", "color_conf": 0.9},
    {"fid": "d2", "xy": [80.0, 0.0], "class": "car", "color": "gray", "color_conf": 0.9},
    {"fid": "d3", "xy": [120.0, 0.0], "class": "car"},  # no color read
]
_ANCHORS = [
    {"fid": "shed_1", "name": "Shed", "xy": [0.0, 10.0], "class": "building"},
]


def _run(ops):
    return run_program([dict(c) for c in _CANDS],
                       [dict(a) for a in _ANCHORS], {"ops": ops})


def _expect_error(ops, *needles):
    try:
        _run(ops)
    except ProgramError as e:
        for n in needles:
            assert n in str(e), (n, str(e))
    else:
        raise AssertionError(f"expected ProgramError containing {needles!r}")


# --- LAZY-ATTACH: emphasis names the column, NO prior find ------------------


def test_emphasis_attaches_color_column_with_no_prior_find():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_blue", "w": 1}, {"key": "distance to shed_1", "w": -1}]}},
    ])
    rows = out["results"]["r1"]
    by_fid = {r["fid"]: r for r in rows}
    # the column attached on every color-carrying row; the colorless row omits it
    assert by_fid["d0"]["color_blue"] == round(G.color_match("blue", "blue", 0.9), 3)
    assert by_fid["d2"]["color_blue"] == round(G.color_match("gray", "blue", 0.9), 3)
    assert "color_blue" not in by_fid["d3"], by_fid["d3"]
    # the d_<fid> relation auto-attached alongside; the sort ran (rank 0 = the
    # blue car, which is both the color match and nearest the shed)
    assert all(isinstance(r.get("distance to shed_1"), (int, float)) for r in rows)
    assert rows[0]["fid"] == "d0", [r["fid"] for r in rows]
    # permutation: no row added or dropped
    assert sorted(r["fid"] for r in rows) == ["d0", "d1", "d2", "d3"]


# --- FIND-ATTACHES-THE-SAME-COLUMN ------------------------------------------


def test_find_color_attaches_the_named_column():
    out = _run([
        {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime", "color": "purple"}},
    ])
    rows = list(out["results"]["r0"])
    by_fid = {r["fid"]: r for r in rows}
    assert by_fid["d1"]["color_purple"] == round(G.color_match("purple", "purple", 0.9), 3)
    assert "color_purple" not in by_fid["d3"]
    # nothing dropped: color annotates, never gates
    assert len(rows) == 4


# --- COEXISTENCE: two color columns on the same rows, independent orders ---


def test_two_color_columns_coexist_and_sort_independently():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_blue", "w": 1}, {"key": "color_purple", "w": 1}]}},
        {"ref": "r2", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_blue", "w": 1}]}},
        {"ref": "r3", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_purple", "w": 1}]}},
    ])
    both = {r["fid"]: r for r in out["results"]["r1"]}
    # BOTH columns ride the same colored rows (the two-color-cue shape)
    for fid in ("d0", "d1", "d2"):
        assert isinstance(both[fid].get("color_blue"), float), both[fid]
        assert isinstance(both[fid].get("color_purple"), float), both[fid]
    # and each single-color sort puts ITS word's car first
    assert out["results"]["r2"][0]["fid"] == "d0"
    assert out["results"]["r3"][0]["fid"] == "d1"


# --- CLOSED-PALETTE: off-palette loud; open word maps on with a note --------


def test_off_palette_word_errors_loud_with_the_palette():
    _expect_error([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_chartreuse", "w": 1}]}},
    ], "closed palette", ", ".join(G.COLOR_ENUM))
    _expect_error([
        {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime", "color": "chartreuse"}},
    ], "closed palette")


def test_open_word_normalization_is_noted():
    out = _run([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_navy", "w": 1}]}},
    ])
    rows = out["results"]["r1"]
    notes = getattr(rows, "notes", []) or []
    assert any("navy" in n and "color_blue" in n for n in notes), notes
    # the canonical column is what rides the rows -- and the sort READ it
    # (the blue car ranks first under the navy cue)
    assert rows[0]["fid"] == "d0", [r["fid"] for r in rows]
    assert isinstance(rows[0].get("color_blue"), float)

    out2 = _run([
        {"ref": "r0", "op": "find",
         "args": {"provenance": "runtime", "color": "navy"}},
    ])
    rows2 = out2["results"]["r0"]
    notes2 = getattr(rows2, "notes", []) or []
    assert any("navy" in n and "color_blue" in n for n in notes2), notes2
    assert isinstance({r["fid"]: r for r in rows2}["d0"].get("color_blue"), float)


# --- OMIT-ON-COLORLESS: all-colorless pool is a loud, truthful error ------


def test_all_colorless_pool_color_emphasis_fails_loud():
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_red", "w": 1}]}},
    ]}
    colorless = [{"fid": f"f{i}", "xy": [float(i), 0.0], "class": "car"}
                  for i in range(3)]
    try:
        run_program(colorless, [], prog)
    except ProgramError as e:
        assert "perceived color" in str(e), str(e)
    else:
        raise AssertionError("expected ProgramError on an all-colorless pool")


# --- ORDERING-ONLY: filter on any color key stays a loud rejection ---------


def test_filter_on_any_color_key_is_rejected():
    for key in ("color_blue", "color_score", "color", "color_conf"):
        _expect_error([
            {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
            {"ref": "r1", "op": "filter_items",
             "args": {"items": "r0", "key": key, "op": "ge", "value": 0.5}},
        ], "ordering signal")


# --- CALIBRATED-UNIT: centered on 0.5, NEVER min-maxed ------------------------


def test_color_columns_are_centered_never_minmaxed():
    rows = [
        {"fid": "a", "color_red": 0.95},
        {"fid": "b", "color_red": 0.05},
    ]
    out = G.score_by(rows, ["color_red"], [1.0])
    # centered terms: v - 0.5 exactly (min-max would re-spread to 1.0 / 0.0)
    assert out[0]["fid"] == "a" and out[0]["score"] == round(0.95 - 0.5, 4)
    assert out[1]["score"] == round(0.05 - 0.5, 4)


# --- CALIBRATED-UNIT: absence is a DIFFERENT FACT on each of the two keys --


def test_an_unseen_colour_is_UNKNOWN_and_ranks_between_match_and_mismatch():
    """A missing ``color_<word>`` means nothing has LOOKED at that row yet."""
    out = G.score_by([{"fid": "match", "color_red": 0.95},
                      {"fid": "unseen"},
                      {"fid": "mismatch", "color_red": 0.05}],
                     ["color_red"], [1.0])
    assert [r["fid"] for r in out] == ["match", "unseen", "mismatch"], out
    assert out[1]["score"] == 0.0, out
    # w < 0 flips which VALUE wins; unknown stays the midpoint
    down = G.score_by([{"fid": "match", "color_red": 0.95},
                       {"fid": "unseen"},
                       {"fid": "mismatch", "color_red": 0.05}],
                      ["color_red"], [-1.0])
    assert [r["fid"] for r in down] == ["mismatch", "unseen", "match"], down


def test_a_missing_name_match_is_still_strictly_worst():
    """The other calibrated key, and its absence is the opposite fact."""
    up = G.score_by([{"fid": "weak", "name_match": 0.05}, {"fid": "absent"}],
                    ["name_match"], [1.0])
    assert [r["fid"] for r in up] == ["weak", "absent"], up
    assert up[1]["score"] < up[0]["score"]
    down = G.score_by([{"fid": "strong", "name_match": 0.95},
                       {"fid": "absent"}], ["name_match"], [-1.0])
    assert [r["fid"] for r in down] == ["strong", "absent"], down


# --- RETIRED-NAME: color_score is gone; the error names the new idiom -------


def test_retired_color_score_key_redirects_to_the_named_columns():
    _expect_error([
        {"ref": "r0", "op": "find", "args": {"provenance": "runtime"}},
        {"ref": "r1", "op": "score_by", "args": {"items": "r0", "emphasis": [
            {"key": "color_score", "w": 1}]}},
    ], "color_<word>")


# --- ONE-VOCABULARY guard: the palette IS the M6 emitted vocabulary ---------


def test_palette_equals_the_m6_emitted_vocabulary():
    from maps.perception.viz.color import _VDW_TO_VOCAB
    assert set(G.COLOR_ENUM) == set(_VDW_TO_VOCAB.values()), (
        G.COLOR_ENUM, sorted(set(_VDW_TO_VOCAB.values())))
    assert G.COLOR_COLUMNS == frozenset("color_" + w for w in G.COLOR_ENUM)
