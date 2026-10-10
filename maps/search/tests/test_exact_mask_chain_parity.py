"""THE exact-mask resolution law + the mega-mask containment idiom.

Every geometry op re-resolves each row's mask BY FID from the authoritative
node registry at compute time (``_row_geom_center``), so chained, sliced,
unioned and re-labeled rows all serve geometry byte-identical to a fresh
compare on the same fids, and a ref compared to ITSELF reads exactly
d_=0 / dc_=0 with both bearings absent (no direction exists).

A ref whose mask contains the whole surface window (``covers_window``,
stamped geometrically at build_surface — no tag test) serves the ONE
containment fact instead of stamping the degenerate-constant d_/brg_gap_
columns; dc_/brg_ stay per-row, and the fact rides the absent-key guards.
"""
from __future__ import annotations

import pytest

from maps.search.analyze_scene.geom_program import (
    ProgramError, _grid_cell, build_surface, run_program)
from maps.search.analyze_scene.text_match import (
    find_score, fuzzy_prepare_query, token_idf)


# --- fixture surface: two mask-bearing buildings in CONTACT, one detached ---
# b1 and b2 share the x=20 edge (d=0, distinct centroids); b3 sits 10 m away
# from b2's right edge. The xy fields are deliberately OFFSET from the mask
# centroids (the tracker's sub-cell point vs the mask's true centroid) so a
# mask-stripped chain that degrades to xy points produces DIFFERENT numbers
# -- exactly the defect class the resolution law kills.
_B1 = {"fid": "prior_1", "name": "One", "xy": [9.0, 9.0],
       "mask": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0], [0.0, 0.0]]],
       "mask_centroid": [10.0, 10.0], "tags": {"building": "yes"}}
_B2 = {"fid": "prior_2", "name": "Two", "xy": [31.0, 11.0],
       "mask": [[[20.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 0.0]]],
       "mask_centroid": [30.0, 10.0], "tags": {"building": "yes"}}
_B3 = {"fid": "prior_3", "name": "Three", "xy": [55.0, 10.0],
       "mask": [[[50.0, 0.0], [60.0, 0.0], [60.0, 20.0], [50.0, 20.0], [50.0, 0.0]]],
       "mask_centroid": [55.0, 10.0], "tags": {"building": "yes"}}
_CAR = {"fid": "detected_0", "xy": [45.0, 10.0], "class": "car"}


def _anchors():
    return [dict(_B1, tags=dict(_B1["tags"])),
            dict(_B2, tags=dict(_B2["tags"])),
            dict(_B3, tags=dict(_B3["tags"]))]


def _geom_cols(row):
    return {k: v for k, v in row.items()
            if k.startswith(("distance to ", "center distance to ",
                             "bearing from ", "facing ",
                             "contains "))}


def test_repeated_compare_byte_parity():
    # The IDENTICAL compare twice in one program serves identical bytes.
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_1"]}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_1"]}},
    ]}
    out = run_program([dict(_CAR)], _anchors(), prog)["results"]
    a = {r["fid"]: _geom_cols(r) for r in out["r1"]}
    b = {r["fid"]: _geom_cols(r) for r in out["r2"]}
    assert a == b


def test_chain_vs_fresh_parity_over_score_by_slice_and_filter():
    # A compare on rows that came through score_by (which auto-attached its
    # own nested compare), a slice, and a filter serves geometry
    # byte-identical to a fresh compare on the same fids.
    chained = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to prior_1", "w": -1}]}},
        {"ref": "r2", "op": "filter_items",
         "args": {"items": "r1", "key": "distance to prior_1", "op": "le",
                  "value": 1000}},
        {"ref": "r3", "op": "compare",
         "args": {"items": "r2[:3]", "refs": ["prior_1", "prior_2"]}},
    ]}
    out = run_program([dict(_CAR)], _anchors(), chained)["results"]
    chained_rows = {r["fid"]: _geom_cols(r) for r in out["r3"]}
    for fid, cols in chained_rows.items():
        fresh = run_program([dict(_CAR)], _anchors(), {"ops": [
            {"ref": "r0", "op": "compare",
             "args": {"items": [fid], "refs": ["prior_1", "prior_2"]}},
        ]})["results"]["r0"][0]
        for k, v in _geom_cols(fresh).items():
            assert cols.get(k) == v, (fid, k, cols.get(k), v)
        # nothing extra either (score is a score_by field, not geometry)
        assert set(cols) == set(_geom_cols(fresh)), (fid, cols)
    # the contact pair stays contact through the chain
    assert chained_rows["prior_2"]["distance to prior_1"] == 0.0
    assert "facing prior_1" not in chained_rows["prior_2"]


def test_self_ref_reads_zero_and_directionless():
    # A ref compared to ITSELF: d_=0, dc_=0, both bearings absent -- through
    # a CHAIN (the mask-stripped path that used to serve dc 0.5 / brg 90).
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to prior_1", "w": -1}]}},
        {"ref": "r2", "op": "compare",
         "args": {"items": "r1", "refs": ["prior_1"]}},
    ]}
    out = run_program([], _anchors(), prog)["results"]
    row = {r["fid"]: r for r in out["r2"]}["prior_1"]
    assert row["distance to prior_1"] == 0.0, row
    assert row["center distance to prior_1"] == 0.0, row
    assert "facing prior_1" not in row, list(row)
    assert "bearing from prior_1" not in row, list(row)


def test_pasted_row_echo_resolves_geometry_by_fid():
    # A model-pasted row literal (fid + degraded/absent fields) computes from
    # the registry's exact mask, not from the pasted xy.
    prog = {"ops": [
        {"ref": "r0", "op": "compare",
         "args": {"items": [{"fid": "prior_2", "xy": [31.0, 11.0]}],
                  "refs": ["prior_1"]}},
    ]}
    out = run_program([], _anchors(), prog)["results"]
    row = out["r0"][0]
    assert row["distance to prior_1"] == 0.0, row              # masks touch
    assert row["center distance to prior_1"] == 20.0, row            # true centroids, not xy


# --- the mega-mask containment idiom ---------------------------------------

def _mega_surface():
    anchors = _anchors()
    mega = {"fid": "prior_9", "name": "Cambridge", "xy": [30.0, 10.0],
            "mask": [[[-100.0, -100.0], [200.0, -100.0], [200.0, 200.0],
                      [-100.0, 200.0], [-100.0, -100.0]]],
            "mask_centroid": [50.0, 50.0],
            "tags": {"boundary": "administrative"}}
    return [dict(_CAR)], anchors + [mega]


class _N:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return None


def test_build_surface_stamps_covers_window_geometrically():
    # A mask containing every surface row position gets the VISIBLE fact; a
    # normal footprint does not. Cell-space: a 40x40-cell blanket over three
    # single-cell nodes (cell_m=2, origin 0).
    blanket = _N(name="Blanket", centroid=[20, 20],
                 cells=[[r, c] for r in range(41) for c in range(41)],
                 tags={"boundary": "administrative"}, class_name="")
    small = _N(name="Small", centroid=[5, 5], cells=[[5, 5]],
               tags={"building": "yes"}, class_name="")
    far = _N(name="Far", centroid=[35, 35], cells=[[35, 35]],
             tags={"building": "yes"}, class_name="")
    cands, anchors = build_surface(
        [], [("prior_1", small), ("prior_2", far), ("prior_9", blanket)],
        (0.0, 0.0), 2.0)
    rows = {a["fid"]: a for a in anchors}
    assert rows["prior_9"].get("covers_window") is True, rows["prior_9"]
    assert "covers_window" not in rows["prior_1"]
    assert "covers_window" not in rows["prior_2"]


def test_covering_ref_serves_containment_fact_not_zero_columns():
    cands, anchors = _mega_surface()
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_9"]}},
    ]}
    # covers_window is stamped at build_surface; stamp it here directly (the
    # geometric stamp has its own test above).
    anchors[-1]["covers_window"] = True
    out = run_program(cands, anchors, prog)["results"]
    rows = {r["fid"]: r for r in out["r1"]}
    for fid, r in rows.items():
        assert "distance to prior_9" not in r, (fid, list(r))
        assert "facing prior_9" not in r, (fid, list(r))
        assert "center distance to prior_9" in r, (fid, list(r))     # centroid vector rides
    notes = list(getattr(out["r1"], "notes", []))
    assert any("contains the whole surface window" in n for n in notes), notes


def test_score_by_on_covering_ref_gap_key_states_the_containment_fact():
    cands, anchors = _mega_surface()
    anchors[-1]["covers_window"] = True
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to prior_9", "w": -1}]}},
    ]}
    with pytest.raises(ProgramError) as ei:
        run_program(cands, anchors, prog)
    msg = str(ei.value)
    assert "contains the whole surface window" in msg, msg
    assert "center distance to prior_9" in msg, msg                  # the working idiom named


def test_covering_row_as_item_omitted_from_sort_head_with_note():
    # As an ITEM a window-covering row holds d_=0 to every ref by containment
    # and would tie rank-0 of every closest-first sort while never being a
    # candidate referent. RENDER-SHAPE law: it is OMITTED from an ordering
    # op's rendered head and NAMED in one truthful note; the raw rN keeps the
    # full set (covers_window riding the row), and a find render (not an
    # ordering op) still shows it.
    cands, anchors = _mega_surface()
    anchors[-1]["covers_window"] = True
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to prior_1", "w": -1}]}},
        {"ref": "r2", "op": "find",
         "args": {"provenance": "prior", "query": "Cambridge"}},
    ]}
    out = run_program(cands, anchors, prog)["results"]
    from maps.search.analyze_scene.geom_program import compact_results
    env = compact_results(out)
    sort_env = env["r1"]
    # omitted from the rendered head...
    assert "prior_9" not in sort_env["fids"], sort_env["fids"]
    assert "prior_9" not in sort_env["rows"], sort_env["rows"]
    # ...named in one truthful note...
    hint = sort_env.get("hint") or ""
    assert "window-covering" in hint and "prior_9" in hint, hint
    # Named the way the executor spells the column: the rename to phrases
    # left 'd_=0' matching no column the model could act on.
    assert "'distance to' = 0" in hint and "containment" in hint, hint
    # ...len is the TRUE set size and the raw rN keeps the full set.
    assert sort_env["len"] == len(out["r1"])
    assert any(isinstance(r, dict) and r.get("fid") == "prior_9"
               for r in out["r1"])
    # a QUERY-find render is untouched: the covering row still renders
    # there, covers_window riding it as a visible fact. (The browse r0 is
    # unordered — census + len, no row head.)
    assert "fids" not in env["r0"] and "rows" not in env["r0"]
    find_env = env["r2"]
    assert "prior_9" in find_env["fids"], find_env["fids"]
    # the header row (the one column-name surface) carries the fact column
    assert "covers_window" in find_env["rows"].splitlines()[0], find_env["rows"]


def test_tag_fat_row_renders_bounded_head():
    tags = {f"name_{i}": f"v{i}" for i in range(40)}
    cell = _grid_cell(tags)
    assert "+28 more tags (full set on the node)" in cell, cell
    assert cell.count("=") == 12, cell
    # a normal row renders every tag, no marker
    assert "more tags" not in _grid_cell({"a": "1", "b": "2"})


# --- degrader pins ----------------------------------------------------------

def test_distance_key_positive_weight_serves_inversion_note():
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "score_by",
         "args": {"items": "r0",
                  "emphasis": [{"key": "distance to prior_1", "w": 1}]}},
    ]}
    out = run_program([], _anchors(), prog)["results"]
    notes = list(getattr(out["r1"], "notes", []))
    assert any("INVERTS its natural direction" in n
               and "farthest" in n for n in notes), notes
    # rank-0 IS the farthest (the sort itself is untouched -- note only)
    assert out["r1"][0]["fid"] == "prior_3", [r["fid"] for r in out["r1"]]
    # the default direction serves NO such note
    prog["ops"][1]["args"]["emphasis"][0]["w"] = -1
    out = run_program([], _anchors(), prog)["results"]
    assert not any("INVERTS" in n
                   for n in getattr(out["r1"], "notes", []) or []), (
        getattr(out["r1"], "notes", []))


def test_semicolon_compound_query_gets_the_or_remedy():
    prog = {"ops": [{"ref": "r0", "op": "find",
                     "args": {"query":
                              "highway=residential;addr_street=Brookside"}}]}
    with pytest.raises(ProgramError) as ei:
        run_program([], _anchors(), prog)
    msg = str(ei.value)
    assert "OR" in msg and "';'" in msg, msg


def test_tag_family_token_hit_outranks_proper_name_substring_fuzz():
    # The 6232 fixture: a bare 'parking' query must rank the real
    # amenity=parking lot ABOVE naptan rows whose proper-name strings merely
    # CONTAIN the token ('Parking End Cottages').
    # (the real rows are UNNAMED: 'Parking End Cottages' rides naptan tag
    # VALUES, so the substring hit is token-bag fuzz, not the name tier)
    q = fuzzy_prepare_query("parking")
    _texts = ("amenity parking surface",
              "naptan commonname parking end cottages highway bus stop")
    idf = token_idf([{"search_text": t, "name": ""} for t in _texts])
    lot = find_score(
        q, _texts[0], None, {"amenity": "parking"}, idf=idf)
    naptan = find_score(
        q,
        _texts[1],
        None,
        {"naptan:CommonName": "Parking End Cottages",
         "highway": "bus_stop"}, idf=idf)
    assert lot >= 0.80, lot
    assert naptan <= 0.75, naptan
    assert lot > naptan, (lot, naptan)
    # the tag-family tier stays BELOW an exact k=v pair hit
    kv = find_score(fuzzy_prepare_query("amenity=parking"),
                    "amenity parking", None, {"amenity": "parking"}, idf=idf)
    assert kv > lot, (kv, lot)




# --- ref-in-mask containment: the sub-window-but-huge container row --------
# A ward-scale polygon CONTAINS prior_1 (the ref) without covering the whole
# surface window (prior_3 sits outside it), so covers_window does NOT fire —
# yet its d_prior_1 reads 0 and it tops any nearest-first sort as if
# adjacent. The d_=0 row must render the containment fact (contains_<ref>
# cell + note); sorts unchanged, nothing dropped (armC5→C7 recurring).
_WARD = {"fid": "prior_9", "name": "Castle Ward", "xy": [15.0, 15.0],
         "mask": [[[-30.0, -30.0], [45.0, -30.0], [45.0, 45.0],
                   [-30.0, 45.0], [-30.0, -30.0]]],
         "mask_centroid": [7.5, 7.5],
         "tags": {"boundary": "political"}}


def _ward_out():
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_1"]}},
    ]}
    anchors = _anchors() + [dict(_WARD, tags=dict(_WARD["tags"]))]
    return run_program([dict(_CAR)], anchors, prog)["results"]


def test_container_row_reads_d_zero_with_contains_fact():
    rows = {r["fid"]: r for r in _ward_out()["r1"]}
    ward = rows["prior_9"]
    assert ward["distance to prior_1"] == 0.0
    assert ward.get("contains prior_1") is True, ward
    assert "facing prior_1" not in ward  # contact: no local direction


def test_touching_neighbor_gets_no_contains_fact():
    # prior_2 shares prior_1's x=20 edge: d_=0 by ADJACENCY — its mask does
    # not contain the ref, so no containment cell.
    rows = {r["fid"]: r for r in _ward_out()["r1"]}
    b2 = rows["prior_2"]
    assert b2["distance to prior_1"] == 0.0
    assert "contains prior_1" not in b2, b2


def test_self_compare_gets_no_contains_fact():
    rows = {r["fid"]: r for r in _ward_out()["r1"]}
    assert "contains prior_1" not in rows["prior_1"]


def test_containment_note_names_the_row_and_the_fact():
    prog = {"ops": [
        {"ref": "r0", "op": "find", "args": {"provenance": "prior"}},
        {"ref": "r1", "op": "compare",
         "args": {"items": "r0", "refs": ["prior_1"]}},
    ]}
    anchors = _anchors() + [dict(_WARD, tags=dict(_WARD["tags"]))]
    out = run_program([dict(_CAR)], anchors, prog)
    notes = out.get("notes") or []
    if not notes:  # notes may ride the rendered envelope instead
        notes = [n for v in (out.get("results") or {}).values()
                 for n in (getattr(v, "notes", None) or [])]
    joined = " ".join(str(n) for n in notes)
    assert "prior_9" in joined and "contains prior_1" in joined, joined
    assert "containment, not adjacency" in joined
