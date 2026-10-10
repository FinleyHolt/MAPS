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
Marks, the margin band and the JPEG come from the SAME overlay renderer the
look crops use (``mllm_overlay.render_look_overlay``); the schematic draws
no ordinal badges. Identity chips: the fid's digits for a prior, ``dN`` for
a detection.

Area/building cell paint is largest feature first so the smallest claim
wins a contested cell (the segmap rule).
"""

from __future__ import annotations

import hashlib
import logging
import re

log = logging.getLogger(__name__)

# Delivered side in pixels. The schematic is drawn ink with no ground
# sampling distance, so it always renders at the full pixel budget. The serve
# applies no max_pixels bound and image tokens scale at 1 per 1024 px, so
# this constant sets the per-map token bill.
MAP_MAX_PX = 1024

_BG_BGR = (28, 28, 28)          # unrecorded space -- dark, honest nothing

# Bump on any change to the pixels a given spec renders to; it rides every
# map record beside its sha256.
MAP_RENDERER_VERSION = 15


def _n_cells(f) -> int:
    cells = f.get("cells")
    return 0 if cells is None else len(cells)


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
    from maps.search.analyze_scene.cell_mesh import cells_of  # noqa: PLC0415

    x0, y0, x1, y1 = (float(v) for v in window)
    cs = float(cell_size_m)
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    # A degenerate grid (a caller without tile metadata) does not clamp:
    # cells are bake-grid coordinates, and the window intersection below is
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
        # One array pass per feature: its cells all take the one colour.
        a = cells_of(f)
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
    from maps.search.analyze_scene.cell_mesh import cells_of  # noqa: PLC0415
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
    # land on top). Roads stroke their centerline; prior buildings with a
    # recorded ``contour`` fill it over their cell paint; detections get a
    # class-colored ``dN`` chip at their in-window cell centroid. Prior
    # buildings carry a digits chip the same way, inside the mask where the
    # pill fits (or marked: see the chip pass), so the text surface's
    # ``prior_NNNN`` rows are findable in the picture; the record counts the
    # unchipped.
    px_per_m = float(W) / max(span, 1e-6)
    road_w = int(min(9, max(2, round(2.5 * px_per_m))))
    win_pad = 2.0 * cell_size_m  # a cell of slack: strokes clip at the edge
    n_chips = 0
    prior_chip_cands: list[tuple[float, float, str, float, float]] = []
    for f in features:
        kind = str(f.get("kind") or "area")
        prov = str(f.get("provenance") or "prebaked")
        if prov != "detected" and kind == "road":
            # One array expression per road: _w2p's arithmetic per vertex,
            # then the same int32 cast.
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
                       for r, c in cells_of(f)]
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
    # Chip glyphs at the badge floor: ~24 px delivered, absolute
    # (mllm_overlay._badge states the constant).
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
    # hatch below: where detection ink lands on a surveyed footprint the
    # overlap hatches in the building tan, so the survey stays visible.
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
            for r, c in cells_of(f):
                cv2.fillPoly(bmask, [_cell_quad(r, c)], 1)

    # Detections, gathered first: their ink bboxes decide which prior chips
    # relocate into a stack (drawn in the detection pass below). The hatch
    # and the stacked chip state co-location, nothing more.
    det_entries: list[dict] = []
    for f in features:
        if str(f.get("provenance") or "") != "detected":
            continue
        cells_inw = [(int(r), int(c)) for r, c in cells_of(f)
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
        """QUEUE one identity chip; every glyph drains after all ink, so a
        label is never under the thing it labels.

        ``ax_``/``ay_`` are the OBJECT's centre, so a glyph moved out of a
        collision still points at what it names. The pill backdrop draws
        only where the ink under the glyph differs from the chip's own fill
        (a chip off its own object: leader, stacked, edge-clamped).
        """
        _glyphs.append((x0_, y0_, pw_, ph_, txt_, th_, fill_,
                        float(x0_ + pw_ / 2.0 if ax_ is None else ax_),
                        float(y0_ + ph_ / 2.0 if ay_ is None else ay_),
                        tag_))

    def _draw_glyphs(queue):
        """Drain queued glyphs: above all ink, and never overlapping.

        A colliding glyph tries the four diagonal offsets around its anchor
        and takes the least crowded; when every offset still overlaps it is
        DROPPED and counted. Earlier entries win (the queue is in priority
        order).
        """
        dropped = 0
        for x0_, y0_, pw_, ph_, txt_, th_, fill_, ax_, ay_, tag_ in queue:
            # Candidate placements: the requested one, then the four
            # diagonal offsets, scored by overlap AREA.
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
                # Every offset still overlaps: count it, and record its tag
                # so the record never claims a chip the picture lacks.
                dropped += 1
                if tag_ is not None:
                    _undrawn.add(tag_)
                continue
            x0_, y0_ = box[0], box[1]
            _placed_boxes.append(box)
            # A moved glyph gets a leader back to its object past ~12 px.
            _gap = ((x0_ + pw_ / 2.0 - ax_) ** 2
                    + (y0_ + ph_ / 2.0 - ay_) ** 2) ** 0.5
            if _gap > 12.0:
                cv2.line(rgb, (int(x0_ + pw_ / 2.0), int(y0_ + ph_ / 2.0)),
                         (int(ax_), int(ay_)), (0, 0, 0), 3, cv2.LINE_AA)
                cv2.line(rgb, (int(x0_ + pw_ / 2.0), int(y0_ + ph_ / 2.0)),
                         (int(ax_), int(ay_)), fill_, 1, cv2.LINE_AA)
            # Sampled after every fill, so the backdrop test sees the final
            # ink.
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

    # Prior-fid chips. WHO gets one is the TEXT side's call: when the request
    # carries marks (the set the model's own query returned), only those
    # objects are chipped, so identity costs |marks|, not |scene|. With no
    # marks the fit rule (the pill fits inside the drawn footprint) decides.
    #
    # A MARKED object is chipped whether or not the pill fits, with a leader
    # line to its footprint when the pill sits outside. A candidate under
    # detection ink STACKS under that detection's dN chip (one per
    # detection). The skipped counts ride the record.
    _marked_fids = {str(mk.get("fid")) for mk in (spec.get("marks") or [])
                    if mk.get("fid")}
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
            # footprint and a line runs back to it, so the binding is drawn
            # rather than inferred from containment.
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

    # Detections LAST: canvas-resolution cell fill (building fills can bury
    # the cell-raster paint underneath; this pass draws over them), the
    # coresidency hatch (45-degree tan stripes over a surveyed footprint),
    # the dN chip, and the stacked prior-fid pill beneath it.
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
            # Queued like every other glyph, so a later fill cannot cut it.
            _chip(sx0, sy0, pw2, ph2, digits, th2, _tan)

    def _proj_poly(boundary):
        """World verts -> the OPEN in-frame arcs (``look_render._proj_poly``'s
        contract): a footprint running past the frame shows the part in
        frame, with gaps at the crossings."""
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
            # digits are 1 = center, 2 = the target when there is one, then
            # the other marks in caller order.
            _base = 2 if target_px is not None else 1
            legend.append({"digit": _base + len(neighbor_px), "fid": mfid,
                           "label": str(mk.get("label") or mfid),
                           "role": "context"})

    # NAMES: the cue is written in names. Marked features carry theirs (a
    # mark's label is its name), and so does EVERY named road in the window;
    # a street is many segments sharing one name, so one label per name
    # bounds the cost at the distinct streets in view.
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

    # IDENTITY FIRST, then names: chips drain before the street labels so a
    # name never takes the pixels an id needed.
    n_prior_unchipped += _draw_glyphs(_glyphs)

    _seen_names: set = set()
    for _px, _py, _lab in _name_cands:
        # One label per NAME, not per segment.
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
            center_poly_px=_proj_poly(spec.get("center_boundary")),
            target_px=target_px,
            target_poly_px=target_poly,
            neighbor_px=neighbor_px,
            neighbor_polys_px=neighbor_polys,
            band_text=band_text,
            neighbor_digit_start=(3 if target_px is not None else 2),
            # a schematic shape is filled ink, not evidence pixels: the
            # digit sits IN the mark
            badge_inside=True,
            # No ordinal badges on a schematic: every object the caller
            # cares about carries its own fid chip, so an ordinal would be a
            # second identity. The legend still rides the record.
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
        # the ids actually READABLE off this image: a glyph the placement
        # could not fit is counted out.
        "prior_chips": [c for c in prior_chips_drawn
                        if ("prior", str(c["fid"])) not in _undrawn],
        "detection_chips": [
            {"fid": de["fid"], "chip": de["chip"],
             "xy": [round(de["cx"], 2), round(de["cy"], 2)]}
            for de in det_entries
            if ("det", str(de["fid"])) not in _undrawn],
        "renderer_version": MAP_RENDERER_VERSION,
        # what the caller handed over, so the record reads back as the
        # request that made it
        "center_boundary": spec.get("center_boundary"),
        "marks": spec.get("marks") or [],
        "center_fid": center_fid,
        "center_label": center_label,
        "center_xy": [round(cx, 2), round(cy, 2)],
        "context_m": round(span, 2),
        "world_window": [round(v, 2) for v in (x0, y0, x1, y1)],
        "gsd_m_per_px": round(span / max(W, 1), 4),
        # the delivered pixel grid
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
