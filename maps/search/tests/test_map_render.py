"""The world-state map render: geometry, palette and honesty contracts.

Pinned here: the north-up flip (grid row 0 = SOUTH edge -> image bottom),
the kind-ranked paint (areas muted / roads stroked / prior buildings tan /
detections class-hued with dN identity chips), the class-colour agreement
between paint and legend word, the rotated band stating its true up, and
the record carrying the re-render spec facts (window, gsd, sha256).
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from maps.perception.viz.osm_palette import (
    osm_kind,
    osm_label,
    paint_color,
    paint_word,
)
from maps.search.render.map_render import (
    features_window_paint,
    render_map_from_spec,
)
from maps.search.tests.fakes import map_mission

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
    assert rec["detections_chipped"] == 1
    base = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [], **_TILE)
    assert rec["sha256"] != base["sha256"]  # the chip + paint are real ink


def test_prior_fid_chips_only_where_the_pill_fits():
    """A surveyed building's fid digits chip IN its mask so the text
    surface's prior_NNNN rows are findable in the picture — but only where
    the pill fits the drawn footprint; the rest are counted, never
    silently dropped."""
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
    assert rec["prior_chips"][0]["chip"] == "4829"


def test_the_centre_carries_no_competing_numeral(monkeypatch):
    """The centre draws its outline and NO ordinal badge.

    Asked which chip named the outlined building, the model answered the
    centre badge "1" on 17 of 18 probe items — id_binding 0/44 across two
    independent arms. Withholding that one numeral took id_binding to
    77.3% and detection_count 64.1% -> 93.5% (the badge was being counted
    among the d-chips). A centre badge that comes back re-opens both.
    """
    b = {"fid": "prior_1", "label": "building", "kind": "building",
         "provenance": "prebaked", "cells": [[4, 4]],
         "contour": [[2.0, 2.0], [18.0, 2.0], [18.0, 18.0], [2.0, 18.0]]}
    d = {"fid": "detected_2", "label": "house", "kind": "detection",
         "provenance": "detected", "cells": [[4, 4]]}
    from maps.perception.viz import mllm_overlay

    seen = []
    real = mllm_overlay.render_look_overlay
    monkeypatch.setattr(mllm_overlay, "render_look_overlay",
                        lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [b, d], **_TILE)
    assert seen and all(kw.get("draw_digits") is False for kw in seen), seen
    chips = ([c["chip"] for c in rec["prior_chips"]]
             + [c["chip"] for c in rec["detection_chips"]])
    assert chips, "no chips drawn — the guard would pass vacuously"
    # The centre legend entry exists (callers read it) but must contribute
    # NO glyph: its digit is withheld at draw time.
    centre = [e for e in rec["legend"] if e.get("role") == "center"]
    assert len(centre) == 1


def test_coresident_detection_hatches_and_stacks_both_chips():
    """A detection sitting on its surveyed twin: the newest knowledge wins
    the pixels, but the overlap stays VISIBLE (tan hatch through the
    detection fill) and BOTH ids survive — the prior's fid chip stacks
    under the detection's dN chip. Co-location is drawn; whether the pair
    is one object stays the graph's re-identification call."""
    prior = {"fid": "prior_4829", "label": "building", "kind": "building",
             "provenance": "prebaked", "cells": [[4, 4], [4, 5], [5, 4],
                                                 [5, 5]],
             "contour": [[6.0, 6.0], [14.0, 6.0], [14.0, 14.0], [6.0, 14.0]]}
    det = {"fid": "detected_3", "label": "house", "provenance": "detected",
           "cells": [[4, 4], [4, 5], [5, 4], [5, 5]]}
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, [prior, det], **_TILE)
    assert rec["detections_chipped"] == 1
    assert rec["priors_chipped"] == 1
    assert rec["chips_stacked"] == 1
    assert rec["coresident_hatched"] == 1
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
                    "is_target": True}]},
        _FEATURES, **_TILE)
    assert rec is not None
    assert rec["kind"] == "map"
    assert rec["world_window"] == [0.0, 0.0, 20.0, 20.0]
    assert rec["sha256"] and rec["jpeg"]
    assert [e["digit"] for e in rec["legend"]] == [1, 2]
    words = {c["label"]: c["color"] for c in rec["classes"]}
    assert words["residential"] == paint_word("residential", kind="area")
    assert words["car"] == paint_word("car", detected=True)
    # the delivered image decodes and is band + square canvas
    img = cv2.imdecode(np.frombuffer(rec["jpeg"], np.uint8),
                       cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] == img.shape[1] + 20


def test_kind_palette_separates_house_street_and_landuse():
    """The digest palette's measured failure: building / residential-road /
    residential-landuse landed on one green (the facing render gate,
    12/12 sheets unreadable). The kind layer must hold them apart."""
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


def test_osm_label_rule_matches_the_dashboard():
    assert osm_label({"highway": "residential"}) == "residential"
    assert osm_label({"building": "yes"}) == "building"
    assert osm_label({}) == "prior"
    assert osm_label(None, "car") == "car"


def test_targetless_map_marks_start_at_digit_2():
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center",
         "marks": [{"fid": "detected_0", "xy": [19.0, 19.0],
                    "boundary": None, "label": "car detected_0",
                    "is_target": False}]},
        _FEATURES, **_TILE)
    assert [e["digit"] for e in rec["legend"]] == [1, 2]
    assert rec["features_painted"] > 0


def test_degenerate_grid_shape_does_not_clamp_the_paint():
    # the live text-only context shipped grid_shape=(0, 0) and every cell
    # was dropped (182/182 dark maps); a degenerate grid must trust cells.
    rgb, counts = features_window_paint(
        _FEATURES, window=(0.0, 0.0, 20.0, 20.0),
        tile_origin_xy=(0.0, 0.0), cell_size_m=2.0, grid_shape=(0, 0))
    assert sum(e["cells"] for e in counts.values()) > 0


def test_the_map_frame_follows_what_was_asked_about():
    """The span used to be a fixed radius or the whole tile whatever the map
    was OF. The default is 6576 m at 1024 px, so a 15 m house rendered 2.3 px
    and a 2 m road 0.31 px -- and this verb runs in 94.5% of episodes. The
    marks were resolved and badged and then ignored by the framing, so
    narrowing to three houses and asking to see them returned the whole tile.

    Framing to the request is 8-18x linear resolution on it, and it makes
    pointing off the picture accurate: the caption carries the world window,
    so m/px is what an eyeballed coordinate costs.
    """
    from maps.search.parse_compile.reasoner_loop import (
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



def test_the_image_axis_convention_is_stated_once_and_correctly():
    """30% of episodes visibly stall on "does Y increase?" -- the caption
    said "north-up" and gave the window, then left the model to infer that
    image row 0 is the north edge. Verbatim from a run:

        "the top of the image should be the larger Y (3641) and the bottom
         the smaller Y (3581)? OR, does Y increase So"

    Drawing tick numbers on the pixels did not move that (30% with, 29%
    without). A convention is stated, not drawn. This is the KEY half of the
    Set-of-Mark pattern the badge legend already measured at +23.3pp: the
    mark works once something binds it to meaning.

    It lives in the system prompt because it is true of every picture and
    the prefix is cached at 99.9%, so saying it once costs nothing.
    """
    from maps.search.parse_compile.reasoner_agent import served_system_text

    # normalized: the prompt wraps, and a line break is not a claim
    t = " ".join(served_system_text().split())
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
    """Map and look records carry the pixel width their jpeg was encoded at."""
    import cv2

    from maps.search.render.look_render import render_candidate_crops
    from maps.search.tests.fakes import ORTHO_MAP

    rec = render_map_from_spec({"center_xy": [10.0, 10.0], "context_m": 20.0,
                                "center_label": "map center"}, [], **_TILE)
    (look,) = render_candidate_crops(
        [{"center_fid": "", "center_xy": [100.0, 100.0], "center_label": "here",
          "center_boundary": None, "annotate": False, "marks": []}],
        map_name=ORTHO_MAP, rgbd_dir=ortho_dir)
    for r in (rec, look):
        img = cv2.imdecode(np.frombuffer(r["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[1] == r["side_px"], (img.shape, r["side_px"])


def test_marks_decide_who_is_chipped_not_footprint_size():
    """Identity follows the TEXT side's selection, and costs |marks|.

    The old rule chipped whatever fit a pill inside its own footprint —
    49.7% of in-frame priors, with which half decided by how large they
    drew. That does not survive a window holding hundreds of detections.
    When the request carries marks, only the marked objects are chipped;
    a marked object is chipped even when the pill CANNOT fit, with a
    leader line, because it was asked for.
    """
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
    assert {c["fid"] for c in unsel["prior_chips"]} == {"prior_11",
                                                       "prior_22"}
    assert unsel["priors_unchipped"] == 1

    # Marks present: ONLY the marked object carries identity, and the
    # unmarked big building loses its chip despite still fitting one.
    sel = render_map_from_spec(
        {**spec, "marks": [{"fid": "prior_22", "xy": [14.5, 14.5]}]},
        feats, **_TILE)
    assert {c["fid"] for c in sel["prior_chips"]} == {"prior_22"}

    # A marked object too small for an inside pill is chipped ANYWAY,
    # with a leader — it was asked for, so it gets its id.
    lead = render_map_from_spec(
        {**spec, "marks": [{"fid": "prior_33", "xy": [0.7, 18.7]}]},
        feats, **_TILE)
    chips = {c["fid"]: c for c in lead["prior_chips"]}
    assert set(chips) == {"prior_33"}
    assert chips["prior_33"].get("leader") is True
    assert lead["sha256"] != sel["sha256"]   # real, different ink


def test_a_label_is_never_buried_and_never_stacked():
    """Every glyph draws above all ink, and never on another glyph.

    Prior chips used to be emitted BEFORE the detection pass painted, so
    a detection fill drew straight over them — "3269" delivered as "3".
    And chips had no collision check between themselves, so two ids could
    pile into an unreadable smudge. Both are identity failures: a label
    you cannot read names nothing.
    """
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

    # Every placed glyph box is disjoint: no id stacked on another.
    boxes = []
    for c in (rec["prior_chips"] + rec["detection_chips"]):
        assert c.get("chip"), "a chip record with no text"
    assert len(rec["detection_chips"]) >= 1


def test_the_record_counts_only_chips_that_actually_drew():
    """A chip the record claims but the picture never drew is a lie.

    Chips are recorded in their own pass and DRAWN later, so a glyph the
    placement cannot fit was still being counted. That overstates the
    caption and, worse, makes a counting ground truth score a CORRECT
    answer wrong: detection_count lost 7 of 30 paired items to it, every
    miss an undercount of 1-2, concentrated on the crowded windows.
    """
    # Many detections crammed into one tiny window: placement must fail
    # for some of them, and the record must agree with the pixels.
    dets = [{"fid": f"detected_{i}", "label": "house", "kind": "detection",
             "provenance": "detected", "cells": [[4 + (i % 3), 4 + (i // 3)]]}
            for i in range(12)]
    rec = render_map_from_spec(
        {"center_xy": [10.0, 10.0], "context_m": 20.0,
         "center_label": "map center"}, dets, **_TILE)
    assert rec["detections_chipped"] == len(rec["detection_chips"]), rec
    assert len(rec["detection_chips"]) <= len(dets)
    # Vacuity: a render that drew nothing would satisfy the equality above.
    assert rec["detection_chips"], "no detection chips drawn at all"


def test_cells_as_arrays_render_the_same_bytes_as_cells_as_lists():
    """The live loop hands node cells through as int arrays and the analysis
    readers hand lists; the painter and the delivered map must not know
    which. Converting 3.9 M cells to a list and back per map was the map's
    whole cost on the Cal City bake."""
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
    assert rec_l["sha256"] == rec_a["sha256"]


def test_a_map_after_a_flight_paints_the_flight_and_a_repeat_paints_the_same():
    """The map flatten is built once per surface version. A flight moves
    the version, so the next map carries the detections it folded; a second
    map under the same version is the same bytes."""
    results, maps = map_mission(["show_map(around=prior_7)", "go_to(prior_7)",
                                 "show_map(around=prior_7)", "show_map(around=prior_7)"])
    assert all("error" not in r for r in results[:4]), results
    assert len(maps) == 3 and all(len(j) == 1 for _, j in maps)
    before, after, again = (j[0] for _, j in maps)
    assert before != after, "the map after the flight did not change"
    assert after == again, "a repeat under one version is not the same bytes"
    assert "car=" in maps[1][0] and "car=" not in maps[0][0]
