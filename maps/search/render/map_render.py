"""The ``perceive{map}`` renderer: the world state as a picture, no imagery.

Draws what the graph KNOWS onto a dark canvas, ranked by the question the
schematic answers (spatial arrangement): AREAS (landuse, water, leisure)
paint as muted ground; ROADS draw as light-gray centerline strokes; prior
BUILDINGS fill tan (crisp ``world_contour`` polygons where the bake has
them, 2 m cells otherwise); DETECTIONS paint in their class's saturated
digest hue and carry a small ``dN`` identity chip so every detection is
individually referenceable from text (N = the digits of its fid). Colors
come from ``osm_palette.paint_color`` and stay stable per (label, kind,
provenance). It shows nothing nobody has recorded, and it needs no
flight: the prior map is tile-wide from the first turn, so a map has no
``obs_span_m`` ceiling (a camera crop selects less of a capture; a map is
not a capture).

Pure + data-in-args (the ``geom_compute`` contract): the caller flattens
the graph to plain feature dicts (``{fid, label, kind, cells,
centerline?, contour?, provenance}``); this module does pixels only.
Marks, numbered badges, the margin band and the JPEG come from the SAME
overlay renderer the look crops use
(``mllm_overlay.render_look_overlay``), so the digit->fid legend contract
is one contract across both instruments. Identity chips live in their own
namespace, separated from legend digits AT THE GLYPH: bare digit = legend
badge, ``pNNNN`` = prior fid, ``dN`` = detection fid.

Area/building cell paint is largest feature first so the smallest claim
wins a contested cell (the segmap rule).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re

log = logging.getLogger(__name__)

# Delivered-side pixel budget (the look renderer's constant, same serving
# math) and the schematic's finest useful sampling.
# The schematic is DRAWN ink, so it has no ground sampling distance to
# floor -- there is no sensor and no capture to select less of, so it
# sizes itself to the pixel budget rather than to a photographic span/GSD
# constraint a map does not have. Contours, strokes
# and chips are all drawn at CANVAS resolution, so the budget is the only
# real limit and spending it is free of artefacts.
# Env-overridable so the delivered size can be SWEPT against accuracy and
# token cost without a code change. It is a real knob: the serve applies no
# max_pixels bound, and image tokens scale linearly at 1 per 1024 px
# (measured 512/768/1024/1280 -> 256/576/1024/1600), so this constant sets
# the per-map token bill directly.
MAP_MAX_PX = int(os.environ.get("SEARCH_MAP_MAX_PX", "1024"))

_BG_BGR = (28, 28, 28)          # unrecorded space -- dark, honest nothing

# v2: kind-ranked paint (areas muted / roads stroked light-gray / prior
# buildings tan / detections class-hued with dN identity chips) replacing
# flat digest fills + achromatic-white detections, and building
# world_contour polygon fill replacing cells-only stair-steps.
# v3: coresidency made visible — detection ink over a surveyed footprint
# hatches in the building tan, and the covered prior's fid chip stacks
# under the detection's dN chip (one per detection) instead of being
# suppressed.
# v15: a colliding glyph MOVES rather than dropping, at the least-crowded
# of four offsets around its object, with a leader back to it past ~12 px
# -- the placement the badge design already worked out. Dropping destroys
# an identity, and v14's larger glyphs collide far more often.
# v14: chip glyphs meet the floor the BADGE design already measured --
# 24 px delivered, one Qwen grid cell. Chips ran 16 px at thickness 1
# against the badge's 24 px at thickness 2, and v9 made them relatively
# smaller still by growing the canvas while font scale stayed absolute.
# v13: every glyph draws ABOVE all ink and never on top of another one.
# Prior chips were emitted before the detection pass painted, so a
# detection fill buried them ("3269" delivered as "3"), and chips had no
# collision check between themselves. All glyphs now queue and drain after
# the last fill, through one shared placement list that the street labels
# share too; a colliding glyph is dropped and counted, never stacked.
# v12: an identity chip draws its PILL only where the pill earns its
# pixels. Both kinds filled with the colour of the ink beneath them, so
# the fill was a no-op and only the black border showed -- the lasso
# defect again. Sampled empirically, so a leader/stacked/edge chip (which
# does NOT sit on its own object) keeps its backdrop.
# v11: EVERY named road in the window is labelled, not only marked ones.
# Road identity is naturally sparse -- a street is many segments sharing
# one name, so deduping by name bounds the cost at the distinct streets in
# view (median 3, max 5 over 40 facing windows). Buildings stay on chips:
# 0.4% of them carry a name at all, against 70.8% of roads.
# v10: marked features carry their NAME. The cue is written in names and
# the picture had none: features flatten to their OSM class, so every
# street drew as an anonymous gray line. Names ride the narrowed set, so
# the cost is |marks|.
# v9: the schematic renders at its full pixel budget. A GSD floor is a
# photographic constraint; drawn ink has none, and the floor was resolving
# a 200 m window at 571 px against a 1024 px budget -- 1.8x linear
# resolution, and chips at ~14 px against a ~10-12 px character floor.
# v8: the schematic lassos only the CENTRE. An outline earns its pixels on
# a PHOTOGRAPH, where it separates an object from photographic clutter; on
# filled ink the fill already IS the mask, so outlining every mark redrew
# the same boundary in high-contrast black-on-white and buried chips under
# a jagged edge. The centre keeps its lasso -- it is the only way to say
# THIS one.
# v7: identity follows the TEXT side's selection. When the request carries
# marks, only marked objects are chipped -- with a leader line when the
# pill cannot sit inside the footprint -- and everything else renders as
# unlabelled context ink. Identity cost becomes |marks| rather than
# |scene|, which is the only allocation that survives hundreds of
# detections; the old fit rule chipped 49.7% of in-frame priors and chose
# which half by how large they drew. No marks = no selection to honour, so
# the fit rule stands.
# v6: the ``p`` prefix is REVERTED. It was added on the theory that
# id_binding failed to a namespace collision; v5 showed the centre ordinal
# was the whole defect, and the prefix then priced out badly on its own:
# 28% of chips lost to the wider pill AND side_of_road 71.4% -> 51.2%,
# because a fatter pill covers the building/road boundary that read needs.
# Chips are bare digits again; the centre ordinal stays suppressed. The
# collision that remains -- marks carry bare-numeral badges too -- is
# UNMEASURED, because the probe renders with no marks; measure it there
# before guarding it here.
# v5: the centre draws its outline WITHOUT an ordinal badge. v4 assumed
# the collision was namespace (bare-numeral chip vs bare-numeral badge)
# and prefixing the chips moved id_binding not at all (0.0% -> 0.0%, 76
# items): the badge was winning on SALIENCE, not by being
# confusable. The subject is identified by its outline plus its own chip.
# v4: prior chips carry a ``p`` prefix. They were bare numerals, distinct
# from the white-on-black legend badges by PILL COLOR only, and the probe
# measured that the model does not make that distinction: asked which chip
# named the outlined building it answered the center badge "1" on 17 of 18
# items (id_binding 0.0%). Every identity
# glyph is prefixed now, so the namespaces cannot collide by styling.
MAP_RENDERER_VERSION = 15


def _n_cells(f) -> int:
    cells = f.get("cells")
    return 0 if cells is None else len(cells)


def _cells_of(f):
    """A feature's cells as an ``(N, 2)`` int array.

    The live loop hands the node's own array through untouched and the
    analysis readers hand a list; converting 3.9 M cells to a list and back
    per map was the map's whole cost on the Cal City bake.
    """
    import numpy as np  # noqa: PLC0415

    cells = f.get("cells")
    if cells is None or len(cells) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(cells, dtype=np.int64).reshape(-1, 2)


def features_window_paint(features, *, tile_origin_xy, cell_size_m,
                          grid_shape, window):
    """Paint the window's grid-cell RGB from plain feature dicts.

    ``features``: [{fid, label, kind?, cells [[row, col], ...],
    provenance}] -- row 0 is the SOUTH edge (the bake's frame). ``window``
    is the world square ``(x0, y0, x1, y1)``. Returns ``(rgb, counts)``
    where ``rgb`` is the north-up (row 0 = max y) window raster at cell
    resolution and ``counts`` maps label -> ``{"cells": n, "kind": k}``
    (kind ``road|building|area|detection`` -- legend fuel that knows which
    paint rule fired). Kind-ranked: areas paint muted, buildings tan,
    detections in their class hue; ROAD cells are counted but not painted
    -- roads deliver as centerline strokes (:func:`render_map_from_spec`),
    and a 2 m stair-line under the stroke would double-draw them.
    """
    import numpy as np  # noqa: PLC0415
    from maps.perception.viz.osm_palette import paint_color  # noqa: PLC0415

    x0, y0, x1, y1 = (float(v) for v in window)
    cs = float(cell_size_m)
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    # A degenerate grid (a caller without tile metadata) must not clamp:
    # the (0, 0) "filler" a text-only context carried dropped EVERY cell
    # on every live render (framestage_1000, 182/182 dark maps). Cells are
    # bake-grid coordinates and trusted; the window intersection below is
    # the real guard.
    _clamp = gh > 0 and gw > 0
    ox, oy = (float(tile_origin_xy[0]), float(tile_origin_xy[1]))
    # window in tile-grid coords (col along x, row along y; row 0 = south)
    c0 = int(np.floor((x0 - ox) / cs))
    r0 = int(np.floor((y0 - oy) / cs))
    wcells = max(1, int(round((x1 - x0) / cs)))
    hcells = max(1, int(round((y1 - y0) / cs)))
    rgb = np.full((hcells, wcells, 3), _BG_BGR[::-1], dtype=np.uint8)  # RGB
    counts: dict[str, int] = {}
    # largest first, so the smallest feature covering a cell wins on top;
    # detections paint after every prior (they are the newest knowledge).
    priors = [f for f in features if f.get("provenance") != "detected"]
    dets = [f for f in features if f.get("provenance") == "detected"]
    priors.sort(key=lambda f: -_n_cells(f))

    def _paint(f, color, *, count_only=False):
        # One array pass per feature. A Python loop over every cell of
        # every feature on the map cost a show_map 3 s on the Cal City bake
        # (3.9 M cells) whatever the window held; a feature's cells all take
        # the one colour, so the order they land in is not a fact.
        a = _cells_of(f)
        if len(a) == 0:
            return 0
        r, c = a[:, 0], a[:, 1]
        keep = np.ones(len(a), dtype=bool)
        if _clamp:
            keep &= (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
        rr, cc = r - r0, c - c0
        keep &= (rr >= 0) & (rr < hcells) & (cc >= 0) & (cc < wcells)
        n = int(keep.sum())
        if n and not count_only:
            rgb[rr[keep], cc[keep]] = color
        return n

    def _count(label, kind, n):
        e = counts.setdefault(label, {"cells": 0, "kind": kind})
        e["cells"] += n
        if kind == "detection":
            e["kind"] = kind  # a class seen live outranks its prior twin

    for f in priors:
        label = str(f.get("label") or "prior")
        kind = str(f.get("kind") or "area")
        n = _paint(f, paint_color(label, kind=kind),
                   count_only=(kind == "road"))
        if n:
            _count(label, kind, n)
    for f in dets:
        label = str(f.get("label") or "detection")
        n = _paint(f, paint_color(label, detected=True))
        if n:
            _count(label, "detection", n)
    # grid row 0 is south; an image's row 0 renders at the top -> flip to
    # the north-up delivery frame every crop/bearing already speaks.
    return np.flipud(rgb).copy(), counts


def render_map_from_spec(spec, features, *, tile_origin_xy, cell_size_m,
                         grid_shape) -> dict | None:
    """Render ONE resolved map view. Pure; returns the record or None.

    ``spec``: {center_xy, context_m, center_fid?, center_label,
    center_boundary?, marks [{fid, xy, boundary, label, is_target}],
    frame_up?, frame_note?} -- the crop-request vocabulary, so the loop's
    resolver and audit persist one shape for both instruments.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from maps.perception.viz.mllm_overlay import render_look_overlay  # noqa: PLC0415
    from maps.perception.viz.osm_palette import (  # noqa: PLC0415
        paint_color,
        paint_word,
    )
    from maps.search.render.look_render import (  # noqa: PLC0415
        frame_up_angle_deg,
        up_bearing_deg,
    )
    from maps.util.polyline import clip_ring_to_box  # noqa: PLC0415

    cxy = spec.get("center_xy")
    try:
        span = float(spec.get("context_m") or 0.0)
    except (TypeError, ValueError):
        span = 0.0
    if not (isinstance(cxy, (list, tuple)) and len(cxy) == 2 and span > 0):
        log.warning("map: malformed spec (center_xy/context_m); no map")
        return None
    cx, cy = float(cxy[0]), float(cxy[1])
    x0, y0 = cx - span / 2.0, cy - span / 2.0
    x1, y1 = cx + span / 2.0, cy + span / 2.0
    cell_rgb, counts = features_window_paint(
        features, tile_origin_xy=tile_origin_xy, cell_size_m=cell_size_m,
        grid_shape=grid_shape, window=(x0, y0, x1, y1))
    side_px = MAP_MAX_PX
    rgb = cv2.resize(cell_rgb, (side_px, side_px),
                     interpolation=cv2.INTER_NEAREST)
    H, W = rgb.shape[:2]

    # Declared-view rotation: same law as the look crops (pixels and ink
    # rotate together; corners introduced by the warp stay background).
    _fup = spec.get("frame_up")
    _rotM = None
    if (isinstance(_fup, (list, tuple)) and len(_fup) == 2
            and (float(_fup[0]) or float(_fup[1]))):
        _a = frame_up_angle_deg(float(_fup[0]), float(_fup[1]))
        if abs(_a) > 1e-9:
            _rotM = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), _a, 1.0)
            rgb = cv2.warpAffine(rgb, _rotM, (W, H),
                                 borderValue=_BG_BGR[::-1])

    def _w2p(wx, wy):
        px = (float(wx) - x0) / span * W
        py = (y1 - float(wy)) / span * H
        if _rotM is not None:
            px, py = (_rotM[0, 0] * px + _rotM[0, 1] * py + _rotM[0, 2],
                      _rotM[1, 0] * px + _rotM[1, 1] * py + _rotM[1, 2])
        return px, py

    def _pts_array(pts):
        """``[[x, y], ...]`` as an (N, 2) float array; a malformed list is
        the empty array (the per-point form skipped those points)."""
        try:
            P = np.asarray(pts, dtype=float)
        except (TypeError, ValueError):
            P = np.zeros((0, 2))
        return P if P.ndim == 2 and P.shape[1] == 2 else np.zeros((0, 2))

    def _w2p_array(P):
        """_w2p over every row of ``P`` at once, in the polylines' (N, 1, 2)
        int32 shape: the same operations in the same order per vertex."""
        px = (P[:, 0] - x0) / span * W
        py = (y1 - P[:, 1]) / span * H
        if _rotM is not None:
            px, py = (_rotM[0, 0] * px + _rotM[0, 1] * py + _rotM[0, 2],
                      _rotM[1, 0] * px + _rotM[1, 1] * py + _rotM[1, 2])
        return np.stack([px, py], axis=1).astype(np.int32)[:, None, :]

    # --- feature ink on the delivered canvas (before the overlay, so marks
    # and badges always land on top). Roads stroke their centerline; prior
    # buildings with a recorded ``contour`` fill it crisply over their cell
    # paint (same tan -- fidelity, not information); detections get a small
    # class-colored ``dN`` chip at their in-window cell centroid, the
    # identity that lets text refer to ONE detection among many. Prior
    # buildings carry a ``pNNNN`` chip the same way (tan pill IN the mask,
    # only where the pill FITS the drawn footprint at this zoom) so the
    # text surface's ``prior_NNNN`` rows are findable in the picture; the
    # caption states how many stayed unchipped.
    #
    # EVERY chip is PREFIXED, and that is load-bearing rather than
    # cosmetic. Bare numerals would be disjoint from the white-on-black
    # legend digits by PILL COLOR alone -- and the model cannot make that
    # distinction: asked which chip names the outlined building, it answers
    # the center legend badge. A
    # namespace separated by styling is not separated. So: bare digit =
    # legend badge, ``p`` = prior fid, ``d`` = detection fid, disjoint by
    # construction at the glyph.
    px_per_m = float(W) / max(span, 1e-6)
    road_w = int(min(9, max(2, round(2.5 * px_per_m))))
    win_pad = 2.0 * cell_size_m  # a cell of slack: strokes clip at the edge
    n_chips = 0
    prior_chip_cands: list[tuple[float, float, str, float, float]] = []
    for f in features:
        kind = str(f.get("kind") or "area")
        prov = str(f.get("provenance") or "prebaked")
        if prov != "detected" and kind == "road":
            # One array expression per road: the same subtract, divide,
            # multiply (and rotation) per vertex as _w2p, then the same
            # int32 cast. 484k centerline vertices on the Cal City bake
            # went through _w2p one at a time on every map.
            P = _pts_array(f.get("centerline") or [])
            if len(P) < 2 or not (
                    (P[:, 0] >= x0 - win_pad) & (P[:, 0] <= x1 + win_pad)
                    & (P[:, 1] >= y0 - win_pad)
                    & (P[:, 1] <= y1 + win_pad)).any():
                continue
            arr = _w2p_array(P)
            cv2.polylines(rgb, [arr], False,
                          paint_color("", kind="road"), road_w,
                          cv2.LINE_AA)
        elif prov != "detected" and kind == "building":
            ring = f.get("contour") or []
            pts = [(float(p[0]), float(p[1])) for p in ring
                   if isinstance(p, (list, tuple)) and len(p) == 2]
            if len(pts) >= 3 and any(
                    x0 - win_pad <= p[0] <= x1 + win_pad
                    and y0 - win_pad <= p[1] <= y1 + win_pad for p in pts):
                arr = np.array([[_w2p(*p)] for p in pts], dtype=np.int32)
                cv2.fillPoly(rgb, [arr],
                             paint_color("", kind="building"), cv2.LINE_AA)
            if not pts:  # cells-only building: chip geometry off its cells
                cs_ = float(cell_size_m)
                tox = float(tile_origin_xy[0])
                toy = float(tile_origin_xy[1])
                pts = [(tox + (int(c) + 0.5) * cs_,
                        toy + (int(r) + 0.5) * cs_)
                       for r, c in _cells_of(f)]
            inw = [p for p in pts if x0 <= p[0] <= x1 and y0 <= p[1] <= y1]
            m = re.search(r"(\d+)$", str(f.get("fid") or ""))
            if inw and m:
                ppx = [_w2p(*p) for p in inw]
                bw = max(p[0] for p in ppx) - min(p[0] for p in ppx)
                bh = max(p[1] for p in ppx) - min(p[1] for p in ppx)
                cx_ = sum(p[0] for p in ppx) / len(ppx)
                cy_ = sum(p[1] for p in ppx) / len(ppx)
                prior_chip_cands.append(
                    (cx_, cy_, m.group(1), bw, bh, str(f.get("fid"))))
    # Chip glyphs at the SAME floor the badge design measured: ~24 px
    # delivered, one Qwen 28-32 px grid cell, frozen absolute and
    # independent of canvas size (mllm_overlay._badge states the constant
    # and its provenance). Chips had been 16 px at thickness 1 against
    # that 24 px at thickness 2 -- below the floor the shared renderer
    # calls measured, and RELATIVELY smaller still once v9 took the canvas
    # to 1024, since font scale is absolute pixels and does not ride the
    # upscale. A chip the model cannot read is clutter.
    _CHIP_FS = 0.9
    _CHIP_TH = 2

    cs_ = float(cell_size_m)
    tox, toy = (float(tile_origin_xy[0]), float(tile_origin_xy[1]))

    def _cell_quad(r, c):
        wx0, wy0 = tox + int(c) * cs_, toy + int(r) * cs_
        return np.array(
            [[_w2p(wx0, wy0)], [_w2p(wx0 + cs_, wy0)],
             [_w2p(wx0 + cs_, wy0 + cs_)], [_w2p(wx0, wy0 + cs_)]],
            dtype=np.int32)

    # Surveyed-building ink mask at canvas resolution, for the coresidency
    # hatch below: where detection ink lands on a surveyed footprint, the
    # overlap hatches in the building tan so the survey stays VISIBLE under
    # the newest knowledge instead of silently buried.
    bmask = np.zeros((H, W), np.uint8)
    for f in features:
        if (str(f.get("provenance") or "prebaked") == "detected"
                or str(f.get("kind") or "area") != "building"):
            continue
        ring = [(float(p[0]), float(p[1])) for p in (f.get("contour") or [])
                if isinstance(p, (list, tuple)) and len(p) == 2]
        if len(ring) >= 3:
            cv2.fillPoly(bmask, [np.array([[_w2p(*p)] for p in ring],
                                          np.int32)], 1)
        else:
            for r, c in _cells_of(f):
                cv2.fillPoly(bmask, [_cell_quad(r, c)], 1)

    # Detections, gathered first: their ink bboxes decide which prior chips
    # relocate into a stack (drawn in the detection pass below). The
    # detection is the newest knowledge and wins the pixels; whether the
    # pair IS one object is the graph's re-identification call, never the
    # renderer's — the hatch + stacked chip state co-location, nothing more.
    det_entries: list[dict] = []
    for f in features:
        if str(f.get("provenance") or "") != "detected":
            continue
        cells_inw = [(int(r), int(c)) for r, c in _cells_of(f)
                     if x0 <= tox + (int(c) + 0.5) * cs_ <= x1
                     and y0 <= toy + (int(r) + 0.5) * cs_ <= y1]
        if not cells_inw:
            continue
        quads = [_cell_quad(r, c) for r, c in cells_inw]
        allpts = np.concatenate(quads)
        m = re.search(r"(\d+)$", str(f.get("fid") or ""))
        det_entries.append({
            "fid": str(f.get("fid") or ""),
            "label": str(f.get("label") or "detection"),
            "quads": quads,
            "chip": (f"d{m.group(1)}" if m else f"d{len(det_entries) + 1}"),
            "cx": float(sum(tox + (c + 0.5) * cs_ for _, c in cells_inw)
                        / len(cells_inw)),
            "cy": float(sum(toy + (r + 0.5) * cs_ for r, _ in cells_inw)
                        / len(cells_inw)),
            "bbox": (float(allpts[:, 0, 0].min()),
                     float(allpts[:, 0, 1].min()),
                     float(allpts[:, 0, 0].max()),
                     float(allpts[:, 0, 1].max())),
            "stacked": [],
        })

    def _covering_detection(bx0_, by0_, bx1_, by1_):
        for de in det_entries:
            d = de["bbox"]
            if bx0_ < d[2] and d[0] < bx1_ and by0_ < d[3] and d[1] < by1_:
                return de
        return None


    _glyphs: list = []

    def _chip(x0_, y0_, pw_, ph_, txt_, th_, fill_, ax_=None, ay_=None,
              tag_=None):
        """QUEUE one identity chip; drawn later, above every fill.

        Queued rather than drawn, because prior chips are emitted before
        the detection pass paints and a detection fill buried them --
        "3269" delivered as "3". Every glyph is drained after all ink, so
        a label is never under the thing it labels.

        ``ax_``/``ay_`` are the OBJECT's centre. A glyph that has to move
        out of a collision still has to point at what it names, so the
        anchor travels with it.

        The PILL draws only where the pill earns its pixels.

        Both chip kinds filled their pill with the colour of the ink they
        sit on -- a prior chip fills tan on a tan building, a detection
        chip fills its own class hue on its own detection -- so the fill
        was a no-op and the pill's whole visible contribution was a black
        rectangle. Same defect the lassos had: a convention built for
        PHOTOGRAPHS, where a glyph needs a controlled backdrop against
        arbitrary pixels, applied to ink whose colour we chose.

        It still earns its pixels where the chip does NOT sit on its own
        object -- a leader-line chip outside its footprint, a stacked
        chip, an edge-clamped one -- so the test is empirical rather than
        case-by-case: sample what is actually under the glyph, and draw
        the backdrop only when it differs.
        """
        _glyphs.append((x0_, y0_, pw_, ph_, txt_, th_, fill_,
                        float(x0_ + pw_ / 2.0 if ax_ is None else ax_),
                        float(y0_ + ph_ / 2.0 if ay_ is None else ay_),
                        tag_))

    def _draw_glyphs(queue):
        """Drain queued glyphs: above all ink, and never overlapping.

        Identity is worthless when two ids sit on top of each other, so a
        glyph whose box collides with one already placed is DROPPED and
        counted rather than stacked into an unreadable pile -- the same
        contract the unchipped count already carries. Earlier entries win,
        and the queue is ordered by priority.
        """
        dropped = 0
        for x0_, y0_, pw_, ph_, txt_, th_, fill_, ax_, ay_, tag_ in queue:
            # MOVE, don't drop. The badge design already solved this: try
            # the four diagonal offsets around the anchor and take the
            # least-crowded, scoring overlap AREA rather than testing a
            # boolean. Dropping destroys an identity; moving keeps it, and
            # a leader restores the binding the move cost. v13 dropped,
            # and v14's larger glyphs collide far more often.
            cands = [(x0_, y0_)]
            for sx_, sy_ in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
                cands.append((ax_ + sx_ * 6 + (0 if sx_ > 0 else -pw_),
                              ay_ + sy_ * 6 + (0 if sy_ > 0 else -ph_)))
            scored = []
            for cx0_, cy0_ in cands:
                cx0_ = int(min(max(cx0_, 1), W - pw_ - 1))
                cy0_ = int(min(max(cy0_, 1), H - ph_ - 1))
                bb = (cx0_, cy0_, cx0_ + pw_, cy0_ + ph_)
                crowd = 0.0
                for ob in _placed_boxes:
                    ix = max(0, min(bb[2], ob[2]) - max(bb[0], ob[0]))
                    iy = max(0, min(bb[3], ob[3]) - max(bb[1], ob[1]))
                    crowd += ix * iy
                scored.append((crowd, bb))
            crowd, box = min(scored, key=lambda c: c[0])
            if crowd > 0:
                # Every offset still overlaps: the frame is genuinely full
                # here, so this one is counted rather than piled on -- and
                # its TAG is recorded, because a chip the record claims but
                # the picture never drew is a lie the caption repeats and
                # a ground truth that scores a correct count wrong.
                dropped += 1
                if tag_ is not None:
                    _undrawn.add(tag_)
                continue
            x0_, y0_ = box[0], box[1]
            _placed_boxes.append(box)
            # A moved glyph gets a leader back to its object, on the same
            # rule the badge uses: bind geometrically past ~12 px.
            _gap = ((x0_ + pw_ / 2.0 - ax_) ** 2
                    + (y0_ + ph_ / 2.0 - ay_) ** 2) ** 0.5
            if _gap > 12.0:
                cv2.line(rgb, (int(x0_ + pw_ / 2.0), int(y0_ + ph_ / 2.0)),
                         (int(ax_), int(ay_)), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.line(rgb, (int(x0_ + pw_ / 2.0), int(y0_ + ph_ / 2.0)),
                         (int(ax_), int(ay_)), fill_, 1, cv2.LINE_AA)
            # Sampled HERE, after every fill, so the backdrop test sees
            # what is really underneath rather than a half-painted canvas.
            cxp = int(min(max(x0_ + pw_ // 2, 0), W - 1))
            cyp = int(min(max(y0_ + ph_ // 2, 0), H - 1))
            under = rgb[cyp, cxp]
            on_own_ink = int(abs(int(under[0]) - fill_[0])
                             + abs(int(under[1]) - fill_[1])
                             + abs(int(under[2]) - fill_[2])) <= 24
            if not on_own_ink:
                cv2.rectangle(rgb, (x0_, y0_), (x0_ + pw_, y0_ + ph_),
                              fill_, -1)
                cv2.rectangle(rgb, (x0_, y0_), (x0_ + pw_, y0_ + ph_),
                              (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb, txt_, (x0_ + 4, y0_ + th_ + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, _CHIP_FS, (0, 0, 0),
                        _CHIP_TH, cv2.LINE_AA)
        return dropped

    _placed_boxes: list = []
    _undrawn: set = set()

    # Prior-fid chips. WHO gets one is the TEXT side's call, not geometry:
    # when the request carries marks (the named set the model's own query
    # returned), only those objects are chipped and everything else stays
    # unlabelled context ink. Identity then costs |marks|, not |scene| --
    # the only rule that survives a window holding hundreds of detections,
    # and the fix for allocating identity by footprint SIZE, which chipped
    # 49.7% of in-frame priors and picked which half by how big they drew.
    # With no marks (a bare "show me the map near X" narrowed nothing)
    # there is no selection to honour, so the fit rule stands as the only
    # identity available.
    #
    # A MARKED object is chipped whether or not the pill fits inside it: it
    # was asked for, so it gets its id, with a leader line to its footprint
    # when the pill has to sit outside. Unmarked chips keep the fit rule --
    # an unrequested id floating off a too-small shape binds to nothing.
    # A candidate under detection ink is not lost: it STACKS under that
    # detection's dN chip (one per detection; the pairing reads off
    # adjacency, so the fit rule does not apply to a stacked pill). The
    # skipped counts ride the record for the caption.
    _marked_fids = {str(mk.get("fid")) for mk in (spec.get("marks") or [])
                    if mk.get("fid")}
    # center_fid is resolved further down for the legend; read it off the
    # spec here rather than reordering the function around one lookup.
    if spec.get("center_fid"):
        _marked_fids.add(str(spec.get("center_fid")))
    _selective = bool(_marked_fids)
    if _selective:
        prior_chip_cands = [t for t in prior_chip_cands
                            if str(t[5]) in _marked_fids]
    n_prior_chips = 0
    n_prior_unchipped = 0
    n_stacked = 0
    prior_chips_drawn: list[dict] = []
    prior_chip_cands.sort(
        key=lambda t: (t[0] - W / 2.0) ** 2 + (t[1] - H / 2.0) ** 2)
    for cx_, cy_, digits, bw, bh, pfid in prior_chip_cands:
        (tw, th), _ = cv2.getTextSize(digits, cv2.FONT_HERSHEY_SIMPLEX,
                                      _CHIP_FS, _CHIP_TH)
        pw, ph = tw + 8, th + 8
        if n_prior_chips + n_stacked >= 25:
            n_prior_unchipped += 1
            continue
        de = _covering_detection(cx_ - pw / 2.0, cy_ - ph / 2.0,
                                 cx_ + pw / 2.0, cy_ + ph / 2.0)
        if de is not None:
            if de["stacked"]:
                n_prior_unchipped += 1  # one stack slot per detection
            else:
                de["stacked"].append(digits)
                prior_chips_drawn.append(
                    {"fid": pfid, "chip": digits, "stacked_under":
                     de["chip"]})
                n_stacked += 1
            continue
        _fits = not (pw > bw + 4 or ph > bh + 4)
        _marked = str(pfid) in _marked_fids
        if not _fits and not _marked:
            n_prior_unchipped += 1
            continue
        if _fits:
            bx0 = int(min(max(cx_ - pw / 2.0, 1), W - pw - 1))
            by0 = int(min(max(cy_ - ph / 2.0, 1), H - ph - 1))
            _lead = None
        else:
            # Outside placement + leader: the pill sits just above the
            # footprint and a line runs back to it, so the binding is
            # DRAWN instead of inferred from containment. Containment is
            # what fails when footprints abut -- the probe caught the
            # model answering a NEIGHBOUR's chip on 2 of 7 items.
            bx0 = int(min(max(cx_ - pw / 2.0, 1), W - pw - 1))
            by0 = int(min(max(cy_ - bh / 2.0 - ph - 4, 1), H - ph - 1))
            _lead = ((bx0 + pw // 2, by0 + ph), (int(cx_), int(cy_)))
        if _lead is not None:
            cv2.line(rgb, _lead[0], _lead[1], (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(rgb, _lead[0], _lead[1],
                     paint_color("", kind="building"), 1, cv2.LINE_AA)
        _chip(bx0, by0, pw, ph, digits, th,
              paint_color("", kind="building"), tag_=("prior", str(pfid)))
        prior_chips_drawn.append(
            {"fid": pfid, "chip": digits,
             **({"leader": True} if _lead is not None else {})})
        n_prior_chips += 1

    # Detections LAST: canvas-resolution cell fill (the cell-raster paint
    # underneath can be buried by building fills; this pass cannot be),
    # the coresidency hatch (45-degree tan stripes where the fill covers a
    # surveyed footprint), the dN identity chip, and the stacked prior-fid
    # pill directly beneath it.
    _tan = paint_color("", kind="building")
    stripe = ((np.add.outer(np.arange(H), np.arange(W)) // 4) % 3 == 0)
    n_hatched = 0
    for de in det_entries:
        det_color = paint_color(de["label"], detected=True)
        cv2.fillPoly(rgb, de["quads"], det_color)
        dmask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(dmask, de["quads"], 1)
        overlap = (dmask > 0) & (bmask > 0)
        if overlap.any():
            rgb[overlap & stripe] = _tan
            n_hatched += 1
        n_chips += 1
        cpx, cpy = _w2p(de["cx"], de["cy"])
        chip = de["chip"]
        (tw, th), _ = cv2.getTextSize(chip, cv2.FONT_HERSHEY_SIMPLEX,
                                      _CHIP_FS, _CHIP_TH)
        bx0 = int(min(max(cpx - (tw + 8) / 2.0, 1), W - tw - 9))
        by0 = int(min(max(cpy - (th + 8) / 2.0, 1), H - th - 9))
        _chip(bx0, by0, tw + 8, th + 8, chip, th, det_color,
              tag_=("det", str(de["fid"])))
        for digits in de["stacked"]:
            (tw2, th2), _ = cv2.getTextSize(digits, cv2.FONT_HERSHEY_SIMPLEX,
                                            _CHIP_FS, _CHIP_TH)
            pw2, ph2 = tw2 + 8, th2 + 8
            sx0 = int(min(max(cpx - pw2 / 2.0, 1), W - pw2 - 1))
            sy0 = int(min(by0 + th + 10, H - ph2 - 1))
            # Queued like every other glyph. Drawn inline it was the LAST
            # thing still painted mid-pass, so a later detection's fill
            # cut it in half -- the "3" of a 3269 chip with the rest under
            # magenta.
            _chip(sx0, sy0, pw2, ph2, digits, th2, _tan)

    def _proj_poly(boundary):
        """World verts -> the OPEN in-frame arcs, the same contract the look
        crops use (``look_render._proj_poly``): the overlay draws arcs, so a
        footprint running past the map frame shows the part in frame and
        leaves the crossings as gaps instead of an edge along the cut."""
        if not (isinstance(boundary, list) and len(boundary) >= 3):
            return None
        pts = [_w2p(p[0], p[1]) for p in boundary
               if isinstance(p, (list, tuple)) and len(p) == 2]
        if len(pts) < 3:
            return None
        return clip_ring_to_box(pts, (-2.0, -2.0, W + 2.0, H + 2.0)) or None

    center_fid = str(spec.get("center_fid") or "")
    center_label = str(spec.get("center_label") or center_fid or "map center")
    legend = [{"digit": 1, "fid": center_fid, "label": center_label,
               "role": "center"}]
    target_px = None
    target_label = ""
    target_poly = None
    neighbor_px: list = []
    neighbor_polys: list = []
    marks_out_of_view: list[str] = []
    _name_cands: list = []
    for mk in spec.get("marks") or []:
        mxy = mk.get("xy") if isinstance(mk, dict) else None
        if not (isinstance(mxy, (list, tuple)) and len(mxy) == 2):
            continue
        px, py = _w2p(mxy[0], mxy[1])
        mfid = str(mk.get("fid") or "")
        if mfid and mfid == center_fid:
            continue
        if not (0.0 <= px <= W and 0.0 <= py <= H):
            if mfid:
                marks_out_of_view.append(mfid)
            continue
        _mlab = str(mk.get("label") or "").strip()
        if _mlab and _mlab != mfid:
            _name_cands.append((px, py, _mlab))
        if bool(mk.get("is_target")) and target_px is None:
            target_px = (px, py)
            target_label = str(mk.get("label") or mfid)
            target_poly = _proj_poly(mk.get("boundary"))
            legend.append({"digit": 2, "fid": mfid, "label": target_label,
                           "role": "target"})
        else:
            neighbor_px.append((px, py))
            neighbor_polys.append(_proj_poly(mk.get("boundary")))
            # digits are 1 = center, then marks in caller order; the +2
            # base holds only while a target occupies digit 2 (the crop
            # overlay's fixed assignment) -- without one, neighbors start
            # AT 2 (the old always-+2 skipped digit 2 on 491/491 renders).
            _base = 2 if target_px is not None else 1
            legend.append({"digit": _base + len(neighbor_px), "fid": mfid,
                           "label": str(mk.get("label") or mfid),
                           "role": "context"})

    # NAMES ON THE NARROWED SET. The cue is written in names -- "the house
    # on Quezon Avenue, east of 85th Street and west of Orchid Drive" --
    # and the picture carried none of them: map features flatten to their
    # OSM CLASS ("residential"), so every street drew as an anonymous gray
    # line and the model could see a road but never WHICH road. Binding the
    # cue meant leaving the image for the row table.
    #
    # A mark's label already IS the name (`_look_label` takes row["name"]),
    # so the name rides the same narrowed set identity does: bounded by
    # |marks|, never by |scene|. Names DESCRIBE -- they order and identify
    # and gate nothing, and every one is an OSM fact the row table already
    # serves, never a harness verdict.
    # EVERY named road in the window, not only the marked ones. Road
    # identity is naturally sparse: a street is many segments sharing one
    # name, so deduping by NAME bounds this at the number of distinct
    # streets in view -- measured median 3, max 5 over 40 facing windows,
    # against dozens of buildings. Without it the model can see that two
    # parallel strokes are different roads but not WHICH is which, and the
    # cue names streets.
    for _f in features:
        if _f.get("kind") != "road" or not _f.get("name"):
            continue
        _cl = _f.get("centerline") or []
        _inw = [p_ for p_ in _cl
                if x0 <= float(p_[0]) <= x1 and y0 <= float(p_[1]) <= y1]
        if not _inw:
            continue
        _mx = sum(float(p_[0]) for p_ in _inw) / len(_inw)
        _my = sum(float(p_[1]) for p_ in _inw) / len(_inw)
        _name_cands.append((*_w2p(_mx, _my), str(_f["name"])))

    # IDENTITY FIRST, then names. Chips drain before the street labels so
    # a name can never take the pixels an id needed -- an unnamed street
    # is still findable by position, an unidentified object is not.
    n_prior_unchipped += _draw_glyphs(_glyphs)

    _seen_names: set = set()
    for _px, _py, _lab in _name_cands:
        # One label per NAME, not per segment. A street is many road
        # features sharing a name, and labelling each drew "Redwood
        # Boulevard" twice in one window.
        if _lab in _seen_names:
            continue
        _seen_names.add(_lab)
        _txt = _lab[:22]
        (_tw, _th), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX,
                                        _CHIP_FS, 1)
        _pw, _ph = _tw + 8, _th + 8
        # Below the mark, nudged up if that would leave the canvas.
        _x0 = int(min(max(_px - _pw / 2.0, 1), W - _pw - 1))
        _y0 = int(min(max(_py + 10, 1), H - _ph - 1))
        _box = (_x0, _y0, _x0 + _pw, _y0 + _ph)
        if any(not (_box[2] < q[0] or q[2] < _box[0]
                    or _box[3] < q[1] or q[3] < _box[1])
               for q in _placed_boxes):
            continue          # overlapping label: drop rather than stack
        _placed_boxes.append(_box)
        cv2.rectangle(rgb, (_x0, _y0), (_x0 + _pw, _y0 + _ph),
                      (250, 250, 250), -1)
        cv2.rectangle(rgb, (_x0, _y0), (_x0 + _pw, _y0 + _ph),
                      (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(rgb, _txt, (_x0 + 4, _y0 + _th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, _CHIP_FS, (0, 0, 0), 1,
                    cv2.LINE_AA)

    if _rotM is None:
        band_text = "N up"
    else:
        band_text = f"up {round(up_bearing_deg(_fup)) % 360} deg"
    try:
        jp = render_look_overlay(
            rgb,
            center_px=_w2p(cx, cy),
            center_label=center_label,
            center_poly_px=_proj_poly(spec.get("center_boundary")),
            px_per_m=float(W) / max(span, 1e-6),
            target_px=target_px,
            target_label=target_label,
            target_poly_px=target_poly,
            neighbor_px=neighbor_px,
            neighbor_polys_px=neighbor_polys,
            band_text=band_text,
            neighbor_digit_start=(3 if target_px is not None else 2),
            # a schematic shape is filled ink, not evidence pixels: the
            # digit sits IN the mark, where the binding is tightest
            badge_inside=True,
            # NO ORDINAL BADGES on a schematic. Every object the caller
            # cares about carries its own fid chip here, so an ordinal is
            # a second identity for the same thing: the centre "1" was
            # answered on 17 of 18 id_binding items, and marked objects
            # got their badge drawn straight over their chip. Outlines
            # still mark; the legend still rides the record for callers.
            draw_digits=False,
            outline_marks=False,
        )
    except Exception as exc:  # noqa: BLE001 -- a render fail skips one map
        log.warning("map: overlay failed at %s: %s", (cx, cy), exc)
        return None
    jp = bytes(jp)
    # legend fuel: label -> the word for what was actually painted (the
    # kind-ranked rule), biggest painted class first
    classes = [{"label": k,
                "color": paint_word(k, kind=e["kind"],
                                    detected=(e["kind"] == "detection")),
                "kind": e["kind"], "cells": e["cells"]}
               for k, e in sorted(counts.items(),
                                  key=lambda kv: -kv[1]["cells"])]
    rec = {
        "jpeg": jp,
        "sha256": hashlib.sha256(jp).hexdigest(),
        "kind": "map",
        "features_painted": int(sum(e["cells"] for e in counts.values())),
        "detections_chipped": int(n_chips) - sum(
            1 for t in _undrawn if t[0] == "det"),
        "priors_chipped": int(n_prior_chips + n_stacked) - sum(
            1 for t in _undrawn if t[0] == "prior"),
        "priors_unchipped": int(n_prior_unchipped),
        "chips_stacked": int(n_stacked),
        "coresident_hatched": int(n_hatched),
        # which ids are actually READABLE off this image — the audit fact a
        # probe or dashboard needs to ask questions only about drawn chips.
        # FILTERED BY WHAT DREW: a glyph the placement could not fit is
        # counted out here, because a chip the record claims and the
        # picture never drew makes the caption overstate and makes a
        # counting ground truth score a CORRECT answer wrong. Measured:
        # detection_count lost 7 of 30 paired items that way, every miss an
        # undercount of 1-2 and concentrated on the crowded windows.
        "prior_chips": [c for c in prior_chips_drawn
                        if ("prior", str(c["fid"])) not in _undrawn],
        "detection_chips": [
            {"fid": de["fid"], "chip": de["chip"],
             "xy": [round(de["cx"], 2), round(de["cy"], 2)]}
            for de in det_entries
            if ("det", str(de["fid"])) not in _undrawn],
        "renderer_version": MAP_RENDERER_VERSION,
        # WHAT THE CALLER HANDED OVER, so the record reads back as the
        # request that made it. The crop record carries both of these and
        # this one did not, so `.get("center_boundary")` answered None and
        # `.get("marks")` answered nothing on EVERY map ever rendered,
        # whatever was passed. A missing-outline diagnosis spent three
        # hypotheses on that silence before noticing it was the RECORD and
        # not the render that had nothing to say.
        "center_boundary": spec.get("center_boundary"),
        "marks": spec.get("marks") or [],
        "center_fid": center_fid,
        "center_label": center_label,
        "center_xy": [round(cx, 2), round(cy, 2)],
        "context_m": round(span, 2),
        "world_window": [round(v, 2) for v in (x0, y0, x1, y1)],
        "gsd_m_per_px": round(span / max(W, 1), 4),
        # the delivered pixel grid: not constant (clamp(span/0.35, 320, 1024)),
        # so the caption has to state it per image
        "side_px": int(W),
        "legend": legend,
        "classes": classes[:10],
        "marks_out_of_view": marks_out_of_view,
    }
    if spec.get("frame_up") is not None:
        rec["frame_up"] = [round(float(spec["frame_up"][0]), 4),
                           round(float(spec["frame_up"][1]), 4)]
        rec["frame_note"] = str(spec.get("frame_note") or "")
    return rec
