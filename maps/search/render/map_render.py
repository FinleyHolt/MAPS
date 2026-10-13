"""The ``show_map`` renderer: the world state drawn as a schematic, no imagery.

Areas paint muted, roads stroke their centerlines, prior buildings fill tan
(their contour, else their 2 m cells), detections paint in their class hue;
colours come from ``osm_palette``. Prior buildings carry their fid digits and
detections a ``dN`` chip, so every drawn object can be named in text. The
caller flattens the graph to feature dicts; marks, band and JPEG come from
``mllm_overlay.render_look_overlay``, without ordinal badges.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Delivered side: ink has no ground sampling distance, so a map is always the
# full 1024 px budget (1 image token per 1024 px).
MAP_MAX_PX = 1024

_BG_BGR = (28, 28, 28)          # unrecorded space
# chip glyphs at the badge floor, ~24 px delivered
_CHIP_FS = 0.9
_CHIP_TH = 2


def _n_cells(f) -> int:
    cells = f.get("cells")
    return 0 if cells is None else len(cells)


def features_window_paint(features, *, tile_origin_xy, cell_size_m,
                          grid_shape, window):
    """``(rgb, counts)`` for the world square ``window``: the north-up cell
    raster painted from feature cells (grid row 0 = south), largest prior
    first so the smallest claim wins a cell, detections last; and label ->
    ``{"cells", "kind"}``. Road cells are counted, not painted: roads are
    stroked by :func:`render_map_from_spec`."""
    import numpy as np  # noqa: PLC0415
    from maps.perception.viz.osm_palette import paint_color  # noqa: PLC0415
    from maps.search.analyze_scene.cell_mesh import cells_of  # noqa: PLC0415

    x0, y0, x1, y1 = (float(v) for v in window)
    cs = float(cell_size_m)
    gh, gw = int(grid_shape[0]), int(grid_shape[1])
    ox, oy = (float(tile_origin_xy[0]), float(tile_origin_xy[1]))
    c0 = int(np.floor((x0 - ox) / cs))
    r0 = int(np.floor((y0 - oy) / cs))
    wcells = max(1, int(round((x1 - x0) / cs)))
    hcells = max(1, int(round((y1 - y0) / cs)))
    rgb = np.full((hcells, wcells, 3), _BG_BGR[::-1], dtype=np.uint8)  # RGB
    counts: dict[str, int] = {}
    priors = [f for f in features if f.get("provenance") != "detected"]
    dets = [f for f in features if f.get("provenance") == "detected"]
    priors.sort(key=lambda f: -_n_cells(f))

    def _paint(f, color, *, count_only=False):
        a = cells_of(f)
        if len(a) == 0:
            return 0
        r, c = a[:, 0], a[:, 1]
        keep = (r >= 0) & (r < gh) & (c >= 0) & (c < gw)
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
            e["kind"] = kind

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
    return np.flipud(rgb).copy(), counts


def render_map_from_spec(spec, features, *, tile_origin_xy, cell_size_m,
                         grid_shape) -> dict | None:
    """Render one map from its spec (``center_xy, context_m, center_fid?,
    center_label, center_boundary?, marks, frame_up?, frame_note?``, the
    look request's vocabulary): the record with its ``jpeg``, or None."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from maps.perception.viz.mllm_overlay import render_look_overlay  # noqa: PLC0415
    from maps.perception.viz.osm_palette import (  # noqa: PLC0415
        paint_color,
        paint_word,
    )
    from maps.search.analyze_scene.cell_mesh import cells_of  # noqa: PLC0415
    from maps.search.render.look_render import (  # noqa: PLC0415
        frame_arcs,
        frame_echo,
        pixel_mapper,
        view_rotation,
        xy_list,
    )

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
    rgb = cv2.resize(cell_rgb, (MAP_MAX_PX, MAP_MAX_PX),
                     interpolation=cv2.INTER_NEAREST)
    H, W = rgb.shape[:2]

    _rotM, band_text = view_rotation(spec.get("frame_up"), W, H)
    if _rotM is not None:
        rgb = cv2.warpAffine(rgb, _rotM, (W, H), borderValue=_BG_BGR[::-1])
    _w2p = pixel_mapper(x0, y1, span, span, W, H, _rotM)
    _tan = paint_color("", kind="building")

    def _in_window(p, pad=0.0):
        return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad

    def _px_ring(pts):
        """World points -> the (N, 1, 2) int32 pixel ring cv2 draws."""
        return np.array([[_w2p(*p)] for p in pts], dtype=np.int32)

    cs_ = float(cell_size_m)
    tox, toy = (float(tile_origin_xy[0]), float(tile_origin_xy[1]))

    def _cell_centre(r, c):
        return tox + (int(c) + 0.5) * cs_, toy + (int(r) + 0.5) * cs_

    def _cell_quad(r, c):
        wx0, wy0 = tox + int(c) * cs_, toy + int(r) * cs_
        return _px_ring([(wx0, wy0), (wx0 + cs_, wy0),
                         (wx0 + cs_, wy0 + cs_), (wx0, wy0 + cs_)])

    def _pill(txt, thick=_CHIP_TH):
        """``(pw, ph, th)``: a chip's pill size and its text height."""
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                      _CHIP_FS, thick)
        return tw + 8, th + 8, th

    def _fit(v, size, lim):
        """A pill's corner coordinate kept a pixel inside the frame."""
        return int(min(max(v, 1), lim - size - 1))

    def _leader(a, b, color, halo):
        cv2.line(rgb, a, b, (0, 0, 0), halo, cv2.LINE_AA)
        cv2.line(rgb, a, b, color, 1, cv2.LINE_AA)

    def _label(txt, bx, by, pw, ph, th, fill, thick):
        """Text on its pill; ``fill`` None draws no pill."""
        if fill is not None:
            cv2.rectangle(rgb, (bx, by), (bx + pw, by + ph), fill, -1)
            cv2.rectangle(rgb, (bx, by), (bx + pw, by + ph), (0, 0, 0), 1,
                          cv2.LINE_AA)
        cv2.putText(rgb, txt, (bx + 4, by + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, _CHIP_FS, (0, 0, 0), thick,
                    cv2.LINE_AA)

    # Feature ink under the overlay: road strokes and building contour
    # fills; each prior with fid digits and a point in the window is a chip
    # candidate (pixel centre x, y, digits, pixel bbox w, h, fid).
    px_per_m = float(W) / max(span, 1e-6)
    road_w = int(min(9, max(2, round(2.5 * px_per_m))))
    win_pad = 2.0 * cell_size_m  # a cell of slack: strokes clip at the edge
    prior_chip_cands: list[tuple[float, float, str, float, float, str]] = []
    # Where detection ink lands on a surveyed footprint (bmask) the overlap
    # hatches tan, so the survey stays visible.
    bmask = np.zeros((H, W), np.uint8)
    for f in features:
        if str(f.get("provenance") or "prebaked") == "detected":
            continue
        kind = str(f.get("kind") or "area")
        if kind == "road":
            try:
                P = np.asarray(f.get("centerline") or [], dtype=float)
            except (TypeError, ValueError):
                continue
            if (P.ndim == 2 and P.shape[1] == 2 and len(P) >= 2 and (
                    (P[:, 0] >= x0 - win_pad) & (P[:, 0] <= x1 + win_pad)
                    & (P[:, 1] >= y0 - win_pad)
                    & (P[:, 1] <= y1 + win_pad)).any()):
                cv2.polylines(rgb, [_px_ring(P)], False,
                              paint_color("", kind="road"), road_w,
                              cv2.LINE_AA)
        elif kind == "building":
            pts = xy_list(f.get("contour"))
            if len(pts) >= 3:
                cv2.fillPoly(bmask, [_px_ring(pts)], 1)
                if any(_in_window(p, win_pad) for p in pts):
                    cv2.fillPoly(rgb, [_px_ring(pts)], _tan, cv2.LINE_AA)
            else:
                for r, c in cells_of(f):
                    cv2.fillPoly(bmask, [_cell_quad(r, c)], 1)
            if not pts:  # cells-only building: chip geometry off its cells
                pts = [_cell_centre(r, c) for r, c in cells_of(f)]
            ppx = [_w2p(*p) for p in pts if _in_window(p)]
            m = re.search(r"(\d+)$", str(f.get("fid") or ""))
            if ppx and m:
                xs, ys = [p[0] for p in ppx], [p[1] for p in ppx]
                prior_chip_cands.append(
                    (sum(xs) / len(ppx), sum(ys) / len(ppx), m.group(1),
                     max(xs) - min(xs), max(ys) - min(ys), str(f.get("fid"))))

    # Detections first: a prior chip under detection ink stacks below its dN.
    det_entries: list[dict] = []
    for f in features:
        if str(f.get("provenance") or "") != "detected":
            continue
        cells_inw = [(r, c) for r, c in cells_of(f)
                     if _in_window(_cell_centre(r, c))]
        if not cells_inw:
            continue
        quads = [_cell_quad(r, c) for r, c in cells_inw]
        allpts = np.concatenate(quads)
        centres = [_cell_centre(r, c) for r, c in cells_inw]
        det_entries.append({
            "label": str(f.get("label") or "detection"),
            "quads": quads,
            # graph fids end in their index: detected_17 chips as d17
            "chip": "d" + re.search(r"(\d+)$", str(f.get("fid"))).group(1),
            "cx": float(sum(p[0] for p in centres) / len(centres)),
            "cy": float(sum(p[1] for p in centres) / len(centres)),
            "bbox": (float(allpts[:, 0, 0].min()),
                     float(allpts[:, 0, 1].min()),
                     float(allpts[:, 0, 0].max()),
                     float(allpts[:, 0, 1].max())),
            "stacked": None,
        })

    # Chips queue as (x, y, pw, ph, text, th, fill, prior fid or None) and
    # draw after all ink.
    glyphs: list = []

    # Prior chips: with marks only the marked objects (a pill that does not
    # fit sits above with a leader); without, every building whose footprint
    # holds the pill. One prior stacks under each covering detection.
    _marked_fids = {str(mk.get("fid")) for mk in (spec.get("marks") or [])
                    if mk.get("fid")}
    if spec.get("center_fid"):
        _marked_fids.add(str(spec.get("center_fid")))
    if _marked_fids:
        prior_chip_cands = [t for t in prior_chip_cands
                            if t[5] in _marked_fids]
    n_prior_chips = n_prior_unchipped = 0
    prior_chip_cands.sort(
        key=lambda t: (t[0] - W / 2.0) ** 2 + (t[1] - H / 2.0) ** 2)
    for cx_, cy_, digits, bw, bh, pfid in prior_chip_cands:
        pw, ph, th = _pill(digits)
        if n_prior_chips >= 25:
            n_prior_unchipped += 1
            continue
        de = next((d for d in det_entries
                   if cx_ - pw / 2.0 < d["bbox"][2]
                   and d["bbox"][0] < cx_ + pw / 2.0
                   and cy_ - ph / 2.0 < d["bbox"][3]
                   and d["bbox"][1] < cy_ + ph / 2.0), None)
        if de is not None:
            if de["stacked"]:
                n_prior_unchipped += 1  # one stack slot per detection
            else:
                de["stacked"] = (digits, pfid)
                n_prior_chips += 1
            continue
        fits = not (pw > bw + 4 or ph > bh + 4)
        if not fits and pfid not in _marked_fids:
            n_prior_unchipped += 1
            continue
        bx0 = _fit(cx_ - pw / 2.0, pw, W)
        if fits:
            by0 = _fit(cy_ - ph / 2.0, ph, H)
        else:
            by0 = _fit(cy_ - bh / 2.0 - ph - 4, ph, H)
            _leader((bx0 + pw // 2, by0 + ph), (int(cx_), int(cy_)), _tan, 2)
        glyphs.append((bx0, by0, pw, ph, digits, th, _tan, pfid))
        n_prior_chips += 1

    # Detections last, over the building fills: cell fill, tan hatch on a
    # surveyed footprint, the dN chip and any stacked prior chip.
    stripe = ((np.add.outer(np.arange(H), np.arange(W)) // 4) % 3 == 0)
    for de in det_entries:
        det_color = paint_color(de["label"], detected=True)
        cv2.fillPoly(rgb, de["quads"], det_color)
        dmask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(dmask, de["quads"], 1)
        overlap = (dmask > 0) & (bmask > 0)
        if overlap.any():
            rgb[overlap & stripe] = _tan
        cpx, cpy = _w2p(de["cx"], de["cy"])
        pw, ph, th = _pill(de["chip"])
        by0 = _fit(cpy - ph / 2.0, ph, H)
        glyphs.append((_fit(cpx - pw / 2.0, pw, W), by0, pw, ph, de["chip"],
                       th, det_color, None))
        if de["stacked"]:
            digits, sfid = de["stacked"]
            pw2, ph2, th2 = _pill(digits)
            glyphs.append((_fit(cpx - pw2 / 2.0, pw2, W),
                           _fit(by0 + th + 10, ph2, H), pw2, ph2, digits,
                           th2, _tan, sfid))

    center_fid = str(spec.get("center_fid") or "")
    center_label = str(spec.get("center_label") or center_fid or "map center")
    legend = [{"digit": 1, "fid": center_fid, "label": center_label,
               "role": "center"}]
    neighbor_px: list = []
    neighbor_polys: list = []
    _name_cands: list = []
    marks_out_of_view: list[str] = []
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
        # every mark is context (a map request names no target): digits
        # are 1 = center, then the marks in caller order
        neighbor_px.append((px, py))
        neighbor_polys.append(frame_arcs(mk.get("boundary"), _w2p, W, H))
        legend.append({"digit": 1 + len(neighbor_px), "fid": mfid,
                       "label": str(mk.get("label") or mfid),
                       "role": "context"})

    # Names: each mark's label and each named road in the window, one label
    # per name, drawn after the chips so a name never takes an id's pixels.
    for _f in features:
        if _f.get("kind") != "road" or not _f.get("name"):
            continue
        _inw = [(float(p_[0]), float(p_[1]))
                for p_ in _f.get("centerline") or []
                if _in_window((float(p_[0]), float(p_[1])))]
        if _inw:
            _name_cands.append((
                *_w2p(sum(p_[0] for p_ in _inw) / len(_inw),
                      sum(p_[1] for p_ in _inw) / len(_inw)),
                str(_f["name"])))

    # Glyphs in queue order, never overlapping: a colliding glyph takes the
    # least crowded diagonal offset (a moved one gets a leader); one with no
    # free offset is dropped and its prior fid recorded. A pill backdrop
    # draws only off the chip's own colour.
    placed: list = []
    undrawn: set = set()
    for gx, gy, pw, ph, txt, th, fill, pfid in glyphs:
        ax_, ay_ = float(gx + pw / 2.0), float(gy + ph / 2.0)
        cands = [(gx, gy)] + [
            (ax_ + sx_ * 6 + (0 if sx_ > 0 else -pw),
             ay_ + sy_ * 6 + (0 if sy_ > 0 else -ph))
            for sx_, sy_ in ((1, -1), (1, 1), (-1, -1), (-1, 1))]
        scored = []
        for qx, qy in cands:
            qx = int(min(max(qx, 1), W - pw - 1))
            qy = int(min(max(qy, 1), H - ph - 1))
            bb = (qx, qy, qx + pw, qy + ph)
            scored.append((sum(
                max(0, min(bb[2], ob[2]) - max(bb[0], ob[0]))
                * max(0, min(bb[3], ob[3]) - max(bb[1], ob[1]))
                for ob in placed), bb))
        crowd, box = min(scored, key=lambda c: c[0])
        if crowd > 0:
            if pfid is not None:
                undrawn.add(pfid)
            continue
        gx, gy = box[0], box[1]
        placed.append(box)
        if ((gx + pw / 2.0 - ax_) ** 2 + (gy + ph / 2.0 - ay_) ** 2) ** 0.5 > 12.0:
            _leader((int(gx + pw / 2.0), int(gy + ph / 2.0)),
                    (int(ax_), int(ay_)), fill, 3)
        under = rgb[int(min(max(gy + ph // 2, 0), H - 1)),
                    int(min(max(gx + pw // 2, 0), W - 1))]
        on_own_ink = sum(abs(int(under[i]) - fill[i]) for i in range(3)) <= 24
        _label(txt, gx, gy, pw, ph, th, None if on_own_ink else fill, _CHIP_TH)

    _seen_names: set = set()
    for _px, _py, _lab in _name_cands:
        if _lab in _seen_names:
            continue
        _seen_names.add(_lab)
        _txt = _lab[:22]
        _pw, _ph, _th = _pill(_txt, 1)
        _x0, _y0 = _fit(_px - _pw / 2.0, _pw, W), _fit(_py + 10, _ph, H)
        _box = (_x0, _y0, _x0 + _pw, _y0 + _ph)
        if any(not (_box[2] < q[0] or q[2] < _box[0]
                    or _box[3] < q[1] or q[3] < _box[1])
               for q in placed):
            continue          # overlapping label: drop rather than stack
        placed.append(_box)
        _label(_txt, _x0, _y0, _pw, _ph, _th, (250, 250, 250), 1)

    try:
        jp = render_look_overlay(
            rgb,
            center_px=_w2p(cx, cy),
            center_poly_px=frame_arcs(spec.get("center_boundary"), _w2p, W, H),
            neighbor_px=neighbor_px,
            neighbor_polys_px=neighbor_polys,
            band_text=band_text,
            # the fid chips are the identity; no ordinal badges
            draw_digits=False,
            outline_marks=False,
        )
    except Exception as exc:  # noqa: BLE001 -- a render fail skips one map
        log.warning("map: overlay failed at %s: %s", (cx, cy), exc)
        return None
    # label -> the colour word painted, biggest class first
    classes = [{"label": k,
                "color": paint_word(k, kind=e["kind"],
                                    detected=(e["kind"] == "detection")),
                "kind": e["kind"], "cells": e["cells"]}
               for k, e in sorted(counts.items(),
                                  key=lambda kv: -kv[1]["cells"])]
    rec = {
        "jpeg": bytes(jp),
        "kind": "map",
        "features_painted": int(sum(e["cells"] for e in counts.values())),
        # a prior chip the placement could not fit is counted out
        "priors_chipped": n_prior_chips - len(undrawn),
        "priors_unchipped": n_prior_unchipped + len(undrawn),
        "center_boundary": spec.get("center_boundary"),
        "marks": spec.get("marks") or [],
        "center_fid": center_fid,
        "center_label": center_label,
        "center_xy": [round(cx, 2), round(cy, 2)],
        "context_m": round(span, 2),
        "world_window": [round(v, 2) for v in (x0, y0, x1, y1)],
        "legend": legend,
        "classes": classes[:10],
        "marks_out_of_view": marks_out_of_view,
    }
    rec.update(frame_echo(spec))
    return rec
