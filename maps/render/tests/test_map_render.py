"""The world-state map render: geometry, palette and honesty contracts."""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from maps.render.osm_palette import (
    osm_kind,
    osm_label,
    paint_color,
    paint_word,
)
from maps.render.map_render import (
    features_window_paint,
    render_map_from_spec,
)
from maps.mission.tests.fakes import map_mission

# A 10x10-cell tile at 2 m/cell, origin (0, 0): a landuse block in the
# SOUTH-WEST corner, a detection cell in the NORTH-EAST corner.
_TILE = {"tile_origin_xy": (0.0, 0.0), "cell_size_m": 2.0,
         "grid_shape": (10, 10)}
_FEATURES = [
    {"fid": "prior_0", "label": "residential", "kind": "area",
     "provenance": "prebaked",
     "cells": [[0, 0], [0, 1], [1, 0], [1, 1]]},          # SW corner
    {"fid": "detected_0", "label": "car", "provenance": "detected",
     "cells": [[9, 9]]},                                   # NE corner
]


def test_paint_flips_to_north_up_and_colors_by_label():
    rgb, counts = features_window_paint(
        _FEATURES, window=(0.0, 0.0, 20.0, 20.0), **_TILE)
    assert rgb.shape == (10, 10, 3)
    # SW corner (grid row 0) must land at the image BOTTOM after the flip
    assert tuple(rgb[9, 0]) == paint_color("residential", kind="area")
    # NE detection lands at the image TOP-RIGHT, in its class's bright hue
    assert tuple(rgb[0, 9]) == paint_color("car", detected=True)
    assert counts == {"residential": {"cells": 4, "kind": "area"},
                      "car": {"cells": 1, "kind": "detection"}}


def test_smallest_feature_wins_a_contested_cell():
    contested = _FEATURES + [
        {"fid": "prior_1", "label": "garage", "kind": "building",
         "provenance": "prebaked", "cells": [[0, 0]]},
    ]
    rgb, _ = features_window_paint(
        contested, window=(0.0, 0.0, 20.0, 20.0), **_TILE)
    assert tuple(rgb[9, 0]) == paint_color("garage", kind="building")
    assert tuple(rgb[9, 1]) == paint_color("residential", kind="area")


def test_road_cells_are_counted_but_stroked_not_painted():
    road = [{"fid": "prior_9", "label": "residential", "kind": "road",
             "provenance": "prebaked", "cells": [[5, c] for c in range(10)],
             "centerline": [[0.0, 11.0], [20.0, 11.0]]}]
    rgb, counts = features_window_paint(
        road, window=(0.0, 0.0, 20.0, 20.0), **_TILE)
    # no cell paint: the raster stays background where the road runs
    assert tuple(rgb[4, 5]) == (28, 28, 28)
    assert counts == {"residential": {"cells": 10, "kind": "road"}}
    # ...but the delivered map draws the stroke
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, road, **_TILE)
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    road_bgr = paint_color("", kind="road")[::-1]
    dist = np.abs(img.astype(int) - np.array(road_bgr, int)).sum(axis=2)
    assert (dist < 30).any(), "road stroke not found in delivered pixels"


def test_building_contour_fills_crisply():
    sq = [{"fid": "prior_3", "label": "building", "kind": "building",
           "provenance": "prebaked", "cells": [[4, 4]],
           "contour": [[6.0, 6.0], [14.0, 6.0], [14.0, 14.0], [6.0, 14.0]]}]
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, sq, **_TILE)
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    b_bgr = paint_color("", kind="building")[::-1]
    dist = np.abs(img.astype(int) - np.array(b_bgr, int)).sum(axis=2)
    # the 8 m square is ~40% of the window's span: a crisp fill covers
    # far more pixels than its single 2 m cell would (~1% of the canvas)
    assert (dist < 30).mean() > 0.05


def test_detection_carries_an_identity_chip():
    det = [{"fid": "detected_17", "label": "house", "provenance": "detected",
            "cells": [[5, 5]]}]
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, det, **_TILE)
    base = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [], **_TILE)
    assert rec["jpeg"] != base["jpeg"]  # the chip + paint are real ink


def test_prior_fid_chips_only_where_the_pill_fits():
    """A surveyed building's fid digits chip IN its mask so the text surface's
    prior_NNNN rows are findable in the picture — but only where the pill fits the
    drawn footprint; the rest are counted, never silently dropped."""
    big = {"fid": "prior_4829", "label": "building", "kind": "building",
           "provenance": "prebaked", "cells": [[4, 4]],
           "contour": [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]]}
    tiny = {"fid": "prior_7", "label": "building", "kind": "building",
            "provenance": "prebaked", "cells": [[9, 0]],
            "contour": [[0.10, 18.10], [0.32, 18.10],
                        [0.32, 18.32], [0.10, 18.32]]}
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [big, tiny], **_TILE)
    assert rec["priors_chipped"] == 1
    assert rec["priors_unchipped"] == 1


def test_the_centre_carries_no_competing_numeral(monkeypatch):
    """The centre draws its outline and NO ordinal badge."""
    b = {"fid": "prior_1", "label": "building", "kind": "building",
         "provenance": "prebaked", "cells": [[4, 4]],
         "contour": [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]]}
    d = {"fid": "detected_2", "label": "house", "kind": "detection",
         "provenance": "detected", "cells": [[4, 4]]}
    from maps.render import mark_overlay

    seen = []
    real = mark_overlay.render_look_overlay
    monkeypatch.setattr(mark_overlay, "render_look_overlay",
                        lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [b, d], **_TILE)
    assert seen and all(kw.get("draw_digits") is False for kw in seen), seen
    assert rec["priors_chipped"], "no chips drawn — the guard would pass vacuously"
    # The centre legend entry exists (callers read it) but must contribute
    # NO glyph: its digit is withheld at draw time.
    centre = [e for e in rec["legend"] if e.get("role") == "center"]
    assert len(centre) == 1


def test_coresident_detection_hatches_and_stacks_both_chips():
    """A detection sitting on its surveyed twin: the newest knowledge wins the pixels,
    but the overlap stays VISIBLE (tan hatch through the detection fill) and BOTH ids
    survive — the prior's fid chip stacks under the detection's dN chip."""
    prior = {"fid": "prior_4829", "label": "building", "kind": "building",
             "provenance": "prebaked", "cells": [[4, 4], [4, 5], [5, 4],
                                                 [5, 5]],
             "contour": [[6.0, 6.0], [14.0, 6.0], [14.0, 14.0], [6.0, 14.0]]}
    det = {"fid": "detected_3", "label": "house", "provenance": "detected",
           "cells": [[4, 4], [4, 5], [5, 4], [5, 5]]}
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [prior, det], **_TILE)
    assert rec["priors_chipped"] == 1
    assert rec["priors_unchipped"] == 0
    # the hatch is real ink: building tan appears INSIDE the detection's
    # fill region (stripes), alongside the detection hue
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    tan_bgr = np.array(paint_color("", kind="building")[::-1], int)
    det_bgr = np.array(paint_color("house", detected=True)[::-1], int)
    tan_px = (np.abs(img.astype(int) - tan_bgr).sum(axis=2) < 30).sum()
    det_px = (np.abs(img.astype(int) - det_bgr).sum(axis=2) < 30).sum()
    assert tan_px > 50 and det_px > 50


def test_render_record_carries_the_contract():
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_fid": "prior_0", "center_label": "residential prior_0",
         "marks": [{"fid": "detected_0", "xy": [19.0, 19.0],
                    "boundary": None, "label": "car detected_0",
                    "is_target": False}]},
        _FEATURES, **_TILE)
    assert rec is not None
    assert rec["world_window"] == [0.0, 0.0, 20.0, 20.0]
    assert rec["jpeg"]
    assert [e["role"] for e in rec["legend"]] == ["center", "context"]
    words = {c["label"]: c["color"] for c in rec["classes"]}
    assert words["residential"] == paint_word("residential", kind="area")
    assert words["car"] == paint_word("car", detected=True)
    # the delivered image decodes and is band + square canvas
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] == img.shape[1] + 20


def test_kind_palette_separates_house_street_and_landuse():
    trio = [paint_color("building", kind="building"),
            paint_color("residential", kind="road"),
            paint_color("residential", kind="area")]
    for i in range(len(trio)):
        for j in range(i + 1, len(trio)):
            d = sum(abs(a - b) for a, b in zip(trio[i], trio[j]))
            assert d > 120, (trio, i, j)
    assert osm_kind({"highway": "residential"}) == "road"
    assert osm_kind({"landuse": "residential"}) == "area"
    assert osm_kind({"building": "yes"}) == "building"


def test_rotated_map_band_states_its_true_up():
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center",
         "frame_up": [1.0, 0.0], "frame_note": "up = east"},
        _FEATURES, **_TILE)
    assert rec is not None
    assert rec["frame_up"] == [1.0, 0.0]
    assert rec["frame_note"] == "up = east"


def test_osm_label_names_the_class_or_falls_back():
    assert osm_label({"highway": "residential"}) == "residential"
    assert osm_label({"building": "yes"}) == "building"
    assert osm_label({}) == "prior"
    assert osm_label(None, "car") == "car"


def test_targetless_map_marks_are_context():
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center",
         "marks": [{"fid": "detected_0", "xy": [19.0, 19.0],
                    "boundary": None, "label": "car detected_0",
                    "is_target": False}]},
        _FEATURES, **_TILE)
    assert rec["legend"] == [{"label": "map center", "role": "center"},
                             {"label": "car detected_0", "role": "context"}]
    assert rec["features_painted"] > 0


def test_the_map_frame_follows_what_was_asked_about():
    from maps.mission.picture_requests import (
        _MAP_FRAME_MARGIN,
        _MAP_FRAME_MIN_M,
    )

    # three houses on one block, ~120 m apart
    pts = [[1000.0, 2000.0], [1060.0, 2000.0], [1060.0, 2080.0]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = max((max(xs) - min(xs)) * _MAP_FRAME_MARGIN,
               (max(ys) - min(ys)) * _MAP_FRAME_MARGIN,
               _MAP_FRAME_MIN_M)
    assert span <= 400.0, span                  # not the 6576 m tile
    assert span >= _MAP_FRAME_MIN_M             # and not tighter than useful
    # at 1024 px that is sub-metre per pixel, so a 15 m house is tens of px
    assert 15.0 / (span / 1024.0) >= 20.0, span


def _far_set_map(around):
    from maps.mission.picture_requests import map_spec, rounded_xy

    rows = {"detected_1": {"fid": "detected_1", "xy": [1000.0, 1000.0]},
            "detected_2": {"fid": "detected_2", "xy": [1010.0, 1000.0]}}
    spec, _ = map_spec(around, list(rows.values()), rows=rows,
                       resolve=lambda t: rows[t]["xy"] if t in rows
                       else rounded_xy(t), frame=(None, ""))
    return spec


def test_a_terminus_mark_is_drawn_at_its_own_row_not_the_newest_of_its_fid():
    """end_N restarts per ends_of call, so a fid lookup finds the newest
    terminus; a set's marks are drawn at the rows the set holds."""
    from types import SimpleNamespace

    from maps.mission.picture_requests import crop_requests, map_spec, rounded_xy

    old_end = {"fid": "end_0", "xy": [10.0, 20.0]}
    newest = {"end_0": [500.0, 500.0]}

    def resolve(t):
        return newest.get(t) or rounded_xy(t)

    spec, skipped = map_spec("[0, 0]", [old_end], rows={}, resolve=resolve, frame=(None, ""))
    assert [(m["fid"], m["xy"]) for m in spec["marks"]] == [("end_0", [10.0, 20.0])], spec
    reqs, _s, _n = crop_requests(
        ["[0, 0]"], ["r5"], rows={}, resolve=resolve,
        carry=SimpleNamespace(read=lambda h: [old_end] if h == "r5" else None),
        stops={}, obs_minor_m=None, obs_span_m=None, frame=(None, ""))
    marks = {m["fid"]: m["xy"] for m in reqs[0]["marks"]}
    assert marks["end_0"] == [10.0, 20.0], marks


def test_a_map_window_always_holds_its_around():
    spec = _far_set_map("[0, 0]")
    (cx, cy), half = spec["center_xy"], spec["context_m"] / 2.0
    assert abs(cx) <= half and abs(cy) <= half, spec


def test_a_map_around_a_point_far_from_the_set_is_the_point_alone():
    from maps.mission.picture_requests import _MAP_FRAME_MIN_M

    spec = _far_set_map("[0, 0]")
    assert spec["center_xy"] == [0.0, 0.0], spec
    assert spec["context_m"] == _MAP_FRAME_MIN_M, spec
    # a centre the set's own frame holds keeps that frame
    near = _far_set_map("[1004, 1000]")
    assert near["center_xy"] == [1005.0, 1000.0], near


def test_the_image_axis_convention_is_stated_once_and_correctly():
    from maps.mission.served_text import served_system_prompt

    # normalized: the prompt wraps, and a line break is not a claim
    t = " ".join(served_system_prompt().split())
    assert "NORTH-UP" in t
    # the direction, stated rather than implied
    assert "y DECREASES as you look down" in t
    # and the caption format the direction is read against
    assert "[x left..right, y bottom..top]" in t
    # This surface states the convention and stops there. The say prompt
    # carried a worked arithmetic example beside it; this one instead tells
    # the model what a read-off position IS good for -- "says WHERE TO LOOK
    # ... and never where the answer is" -- because only a row can be flown
    # to, so the arithmetic never has to be exact.
    assert "never where the answer is" in t


def test_a_picture_states_its_own_pixel_grid(ortho_dir):
    import cv2

    from maps.render.look_render import render_candidate_crops
    from maps.mission.tests.fakes import ORTHO_MAP

    (look,) = render_candidate_crops(
        [{"center_fid": "", "center_xy": [100.0, 100.0], "center_label": "here",
          "center_boundary": None, "marks": []}],
        map_name=ORTHO_MAP, rgbd_dir=ortho_dir)
    img = cv2.imdecode(np.frombuffer(look["jpeg"], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == look["side_px"], (img.shape, look["side_px"])


def test_marks_decide_who_is_chipped_not_footprint_size():
    """Identity follows the TEXT side's selection, and costs |marks|."""
    def _b(fid, x0, y0, x1, y1):
        return {"fid": fid, "label": "building", "kind": "building",
                "provenance": "prebaked",
                "cells": [[int(y0 // 2), int(x0 // 2)]],
                "contour": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}

    big_a = _b("prior_11", 2.0, 2.0, 9.0, 9.0)
    big_b = _b("prior_22", 11.0, 11.0, 18.0, 18.0)
    tiny = _b("prior_33", 0.10, 18.10, 0.32, 18.32)
    feats = [big_a, big_b, tiny]
    spec = {"center_xy": [10.0, 10.0], "context_m": 20.0,
            "center_label": "map center"}

    # No marks: the fit rule stands — both big ones chip, the tiny one not.
    unsel = render_map_from_spec(dict(spec), feats, **_TILE)
    assert unsel["priors_chipped"] == 2
    assert unsel["priors_unchipped"] == 1

    # Marks present: ONLY the marked object carries identity, and the
    # unmarked big building loses its chip despite still fitting one.
    sel = render_map_from_spec(
        {**spec, "marks": [{"fid": "prior_22", "xy": [14.5, 14.5]}]},
        feats, **_TILE)
    assert sel["priors_chipped"] == 1

    # A marked object too small for an inside pill is chipped ANYWAY,
    # with a leader — it was asked for, so it gets its id.
    lead = render_map_from_spec(
        {**spec, "marks": [{"fid": "prior_33", "xy": [0.7, 18.7]}]},
        feats, **_TILE)
    assert lead["priors_chipped"] == 1
    assert lead["jpeg"] != sel["jpeg"]   # real, different ink


def test_a_label_is_never_buried_and_never_stacked():
    """Every glyph draws above all ink, and never on another glyph."""
    # A building whose chip lands exactly under a detection painted after
    # it — the burial case.
    b = {"fid": "prior_4242", "label": "building", "kind": "building",
         "provenance": "prebaked", "cells": [[4, 4]],
         "contour": [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]]}
    d = {"fid": "detected_9", "label": "house", "provenance": "detected",
         "kind": "detection", "cells": [[r, c] for r in range(3, 8)
                                        for c in range(3, 8)]}
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [b, d], **_TILE)
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    # The detection's own ink covers the middle; a surviving glyph means
    # dark text pixels sit ON that ink rather than under it.
    det_bgr = np.array(paint_color("house", detected=True)[::-1], int)
    mid = img[H // 3: 2 * H // 3, W // 3: 2 * W // 3].astype(int)
    on_det = (np.abs(mid - det_bgr).sum(axis=2) < 40).mean()
    ink = (mid.sum(axis=2) < 200).mean()          # near-black = glyph ink
    assert on_det > 0.2, "the detection did not paint where expected"
    assert ink > 0.0, "no glyph ink over the detection — a label was buried"


def test_the_record_counts_only_chips_that_actually_drew():
    """A chip the record claims but the picture never drew is a lie."""
    # Twelve marked buildings crammed into a few pixels of a wide window:
    # placement must fail for some, and those count as unchipped.
    bs = [{"fid": f"prior_{100 + i}", "label": "building", "kind": "building",
           "provenance": "prebaked", "cells": [[4 + (i % 3), 4 + (i // 3)]]}
          for i in range(12)]
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 400.0,
         "center_label": "map center",
         "marks": [{"fid": b["fid"], "xy": [9.0, 9.0]} for b in bs]},
        bs, **_TILE)
    assert rec["priors_chipped"] + rec["priors_unchipped"] == len(bs), rec
    assert 0 < rec["priors_chipped"] < len(bs), rec


def test_cells_as_arrays_render_the_same_bytes_as_cells_as_lists():
    """The live loop hands node cells through as int arrays and the analysis readers
    hand lists; the painter and the delivered map must not know which."""
    as_lists = _FEATURES + [
        {"fid": "prior_3", "label": "building", "kind": "building",
         "provenance": "prebaked", "cells": [[4, 4], [4, 5]],
         "contour": [[8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0]]},
        {"fid": "prior_9", "label": "residential", "kind": "road",
         "provenance": "prebaked", "cells": [[5, c] for c in range(10)],
         "centerline": [[0.0, 11.0], [20.0, 11.0]]},
    ]
    as_arrays = [{**f, "cells": np.asarray(f["cells"], dtype=np.int32)}
                 for f in as_lists]
    win = dict(window=(0.0, 0.0, 20.0, 20.0), **_TILE)
    rgb_l, counts_l = features_window_paint(as_lists, **win)
    rgb_a, counts_a = features_window_paint(as_arrays, **win)
    assert np.array_equal(rgb_l, rgb_a)
    assert counts_l == counts_a
    assert counts_l["residential"]["cells"] == 14      # vacuity floor
    spec = {"center_xy": [10.0, 10.0], "context_m": 20.0,
            "center_label": "map center",
            "marks": [{"fid": "detected_0", "xy": [19.0, 19.0],
                       "boundary": None, "label": "car",
                       "is_target": False}]}
    rec_l = render_map_from_spec(spec, as_lists, **_TILE)
    rec_a = render_map_from_spec(spec, as_arrays, **_TILE)
    assert rec_l["jpeg"] == rec_a["jpeg"]
    assert rec_l["jpeg"] == rec_a["jpeg"]


def test_a_map_after_a_flight_paints_the_flight_and_a_repeat_paints_the_same():
    """The map flatten is built once per surface version."""
    from maps.mission.tests.fakes import GraphNode

    house = {"prior_9": GraphNode((20.0, 20.0), "house", provenance="prebaked")}
    results, maps = map_mission(["show_map(around=prior_7)", "go_to(prior_7)",
                                 "show_map(around=prior_7)", "show_map(around=prior_7)"],
                                extra_priors=house, spread=True)
    assert all("error" not in r for r in results[:4]), results
    assert len(maps) == 3 and all(len(j) == 1 for _, j in maps)
    before, after, again = (j[0] for _, j in maps)
    assert before != after, "the map after the flight did not change"
    assert after == again, "a repeat under one version is not the same bytes"
    assert "car=" in maps[1][0] and "car=" not in maps[0][0]


def test_dropped_detection_chips_are_not_counted_as_unchipped_buildings():
    """Only prior chips count toward the caption's building tally: dN chips
    that cannot all fit leave it at zero when no building is in frame."""
    dets = [{"fid": f"detected_{i}", "label": "car", "provenance": "detected",
             "cells": [[50, 50]]} for i in range(12)]
    rec = render_map_from_spec(
        {"center_xy": [101.0, 101.0], "context_m": 60.0}, dets,
        tile_origin_xy=(0.0, 0.0), cell_size_m=2.0, grid_shape=(200, 200))
    assert rec["features_painted"] == 12
    assert (rec["priors_chipped"], rec["priors_unchipped"]) == (0, 0)


def test_a_map_around_an_absent_fid_names_it():
    from maps.mission.tests.fakes import drive, run, tool_results
    _r, fr, _f, _d = drive([[run('show_map(around="detected_7")\n')], "done"])
    assert tool_results(fr)[0]["error"] == "detected_7 is not on the live surface"


@pytest.mark.parametrize("code", ['show_map()\n', 'x = None\nshow_map(around=x)\n'])
def test_a_map_around_none_is_refused_as_no_place(code):
    from maps.mission.tests.fakes import drive, run, tool_results
    _r, fr, _f, _d = drive([[run(code)], "done"])
    err = tool_results(fr)[0]["error"]
    assert "around= got None, which is no place" in err, err


def test_a_map_is_drawn_when_the_set_in_scope_cannot_re_derive():
    """The set in scope only badges the map; its refused re-derive is said
    in the map's notes, never the map's own refusal."""
    from maps.mission.tests.fakes import ROADS, grow_drive, run
    res, _g, _a = grow_drive([
        [run('g = find("Long Road").order(nearest="prior_7")\n')],
        [run('go_to("prior_1")\n')],          # the flight removes prior_7
        [run('show_map(around="prior_1")\n')], "done"],
        flights_add=False, evict_on_flight={0: ["prior_7"]}, extra_priors=ROADS)
    out = res[2]
    assert "error" not in out and out["map"]["maps"] == 1, out
    assert any(n.startswith("g cannot be re-derived") for n in out["map"]["ref_notes"]), out
