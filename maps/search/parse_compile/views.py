"""What a look and a map put in front of the model: the render requests they
resolve off the surface rows and the captions that ride their pictures."""

from __future__ import annotations

import math

from maps.query.spatial_math import cell_centre_xy
from maps.search.parse_compile.code_surface import HANDLE_RE, row_token
from maps.render.look_render import (
    LOOK_MAX_BADGES,
    look_label,
    look_outline,
    look_xy,
    up_bearing_deg,
)
from maps.search.withhold import withheld
from maps.xy_token import parse_xy_token

# The server's per-request image cap.
LOOK_MAX_CROPS = 16

# A framed map's slack around what was asked about, and its tightest frame
# (below it the renderer's GSD floor binds and a tighter frame only removes
# context).
_MAP_FRAME_MARGIN = 1.6
_MAP_FRAME_MIN_M = 200.0


def _legend_line(entries: list[dict]) -> str:
    return " ; ".join(
        f"{e['digit']} = {e['label']} ({e['role']})" for e in entries)


def _window_caption(c: dict) -> str:
    """One rendered image's window: span, orientation and world extent."""
    # A rotated image states its true up; only an unrotated one says north-up.
    _fu = c.get("frame_up")
    if isinstance(_fu, (list, tuple)) and len(_fu) == 2:
        _fn = str(c.get("frame_note") or "").strip()
        _deg = round(up_bearing_deg(_fu)) % 360
        _up = (f"up {_deg} deg" + (f" -- {_fn}" if _fn else "")
               # a turn off the square leaves the frame's corners empty
               + ("; its corners lie outside the turned window and render dark"
                  if _deg % 90 else ""))
    else:
        _up = "north-up"
    _cm = c.get("context_m")
    _span = (f" ({_cm:g} m window; {_up})"
             if isinstance(_cm, (int, float)) else f" ({_up})")
    # The numeric extent makes a pixel position back-projectable.
    _ww = c.get("world_window")
    if isinstance(_ww, (list, tuple)) and len(_ww) == 4:
        _span += (f" [x {_ww[0]:g}..{_ww[2]:g}, "
                  f"y {_ww[1]:g}..{_ww[3]:g}]")
    return _span


def look_vision_text(crops: list[dict]) -> str:
    # The digit -> fid binding lives in this text legend; color is left to
    # the pixels, and each crop line states its delivered span.
    def _subject(c):
        return str(c.get("center_fid") or next(
            (e["label"] for e in c.get("legend") or []
             if e.get("role") == "center" and e.get("label")), "?"))

    order = ", ".join(_subject(c) for c in crops)
    # GSD from the crop records themselves, never a literal.
    _gsds = {round(float(c["gsd_m_per_px"]), 2)
             for c in crops if c.get("gsd_m_per_px")}
    _gsd = (f"{min(_gsds):.2f} m/px" if len(_gsds) == 1
            else f"{min(_gsds):.2f}-{max(_gsds):.2f} m/px" if _gsds
            else "m/px unknown")
    # Resampling adds pixels, never detail: the source imagery's GSD bounds
    # what the caption may claim.
    _src = max((float(c["source_gsd_m"]) for c in crops
                if c.get("source_gsd_m")), default=0.0)
    _detail = (f"{_gsd}; detail limited by {_src:.2f} m source imagery"
               if _src and _src > min(_gsds or {_src}) else _gsd)
    base = (f"Overhead crops ({_detail}, nadir), one per fid "
            f"in order: [{order}]. White outline marks, numbered badges; "
            f"per-crop badge legend. Altitude does NOT change crop detail: "
            f"a crop is a digital crop of imagery already captured, so "
            f"flying lower buys none. What it buys is a NEW view square "
            f"somewhere else:")
    for k, c in enumerate(crops):
        entries = c.get("legend") or []
        if not entries:
            continue
        line = _legend_line(entries)
        _span = _window_caption(c)
        # The pixel grid, per image: with the window it converts what the
        # model sees into a world point.
        _px = c.get("side_px")
        if _px:
            _span += f" ({int(_px)}x{int(_px)} px)"
        base += f"\n  crop {k + 1}{_span}: {line}"
        _sxy = c.get("world_xy") or ["?", "?"]
        if c.get("subject_in_window") is False:
            _sw = c.get("world_window") or ["?"] * 4
            base += (" ; WARNING: neither this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] nor any part of its "
                     f"footprint is inside this crop's window "
                     f"[x {_sw[0]}..{_sw[2]}, y {_sw[1]}..{_sw[3]}] "
                     "(edge-clamped or off-tile) -- the subject is not "
                     "in this image; do not read color or arrangement "
                     "from it (the position and window bounds here say "
                     "where it sits relative to this crop).")
        elif c.get("subject_centroid_in_window") is False:
            base += (" ; note: this fid's recorded position "
                     f"[{_sxy[0]}, {_sxy[1]}] is outside this crop's "
                     "window (edge-clamped at the tile boundary), but "
                     "part of its footprint IS in frame -- the outline/"
                     "circle marks the in-frame part.")
        elif c.get("subject_on_data") is False and c.get("source") == "run":
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but this run has not imaged that ground yet "
                     "-- the black region is unobserved, not a dark "
                     "object; its color/shape cannot be read until a leg "
                     "flies over it (go_to it, then look).")
        elif c.get("subject_on_data") is False:
            base += (" ; WARNING: this fid's position "
                     f"[{_sxy[0]}, {_sxy[1]}] is inside this crop's "
                     "window but past the map raster's DATA edge -- "
                     "the black region is off-raster no-data, not a "
                     "dark object; the raster holds no pixels for "
                     "this subject, so its color/shape cannot be "
                     "read from this image.")
        # A subject bigger than the frame: state its size, the ink is cut.
        _ext = c.get("subject_extent_m")
        if c.get("subject_outline_cut") and isinstance(_ext, (list, tuple)):
            base += (f" ; note: this fid's footprint is {_ext[0]:g} x "
                     f"{_ext[1]:g} m and runs past this window, so the "
                     "outline marks only where its edge crosses the frame "
                     "-- not a shape you can read whole here. Its size is "
                     "the fact; the ink is a fragment.")
        _vf = c.get("window_valid_frac")
        if (isinstance(_vf, (int, float)) and _vf < 0.99
                and c.get("subject_on_data") is not False):
            base += (f" ; note: {round((1 - _vf) * 100)}% of this "
                     + ("window is ground this run has not imaged yet "
                        "(black there = unobserved, not shadow)."
                        if c.get("source") == "run" else
                        "window lies past the map raster's data edge "
                        "(black there = no data, not shadow)."))
        if int(c.get("marks_dropped") or 0):
            base += (f" ; +{int(c['marks_dropped'])} requested mark(s) "
                     f"not drawn (badge cap {LOOK_MAX_BADGES}/crop -- "
                     f"request them in another look())")
        _mof = c.get("marks_out_of_view") or []
        if _mof:
            base += (f" ; requested mark(s) {', '.join(_mof)} lie "
                     f"outside this crop's frame and are NOT drawn "
                     f"(absent from the image, not unlabeled)")
        _muf = c.get("marks_unbadged_in_frame") or []
        if _muf:
            base += (f" ; requested mark(s) {', '.join(_muf)}: part of "
                     f"the footprint IS in frame but the centroid badge "
                     f"position falls outside it, so no badge is drawn "
                     f"(in the image, unlabeled)")
    return base


def rounded_xy(p) -> list[float] | None:
    """An ``[x, y]`` token or point -> the point at centimetre precision,
    else None."""
    if isinstance(p, str):
        p = parse_xy_token(p)
    return [round(float(p[0]), 2), round(float(p[1]), 2)] if p else None


def _mark(fid: str, row, xy, is_target: bool) -> dict:
    """One drawn mark: the row's point, outline and badge label."""
    return {"fid": fid, "xy": xy, "boundary": look_outline(row),
            "label": look_label(fid, row, is_target=is_target),
            "is_target": is_target}


def crop_requests(fids: list[str], marks: list | None, *, rows: dict, resolve,
                  carry, stops: dict, obs_minor_m: float | None,
                  obs_span_m: float | None,
                  frame: tuple) -> tuple[list[dict], list[str], list[str]]:
    """One look's fids -> per-crop render requests off ``rows``: each crop
    centred on the place ``resolve`` names, badged as the target, with
    ``marks`` (places or handles) drawn beside it, capped at
    :data:`LOOK_MAX_CROPS`.
    ``(requests, skipped_fids, ref_notes)``."""
    skipped: list[str] = []
    requests: list[dict] = []
    ref_notes: list[str] = []
    # PIXELS ARE EARNED BY FLYING: a crop centred on ground no flight's
    # inscribed circle (half the footprint's short side) has covered is
    # refused.
    uncovered: list[str] = []

    def _covered(_xy) -> bool:
        if not (isinstance(obs_minor_m, (int, float)) and obs_minor_m > 0):
            return True
        _r = float(obs_minor_m) / 2.0
        return any(math.dist(_xy, _p) <= _r for _p in stops.values())

    # A handle is a legal mark (a street is its ways), each row placed by
    # its own token (a terminus by its xy, its end_N being per call).
    own: dict = {}
    for t in marks or []:
        if HANDLE_RE.fullmatch(t):
            for _r in carry.read(t) or []:
                if _r.get("fid"):
                    own.setdefault(str(_r["fid"]), row_token(_r))
        else:
            own.setdefault(t, t)
    for _i, f in enumerate(fids):
        if len(requests) >= LOOK_MAX_CROPS:
            ref_notes.append(
                f"only the first {LOOK_MAX_CROPS} crops of this call "
                f"rendered ({len(fids) - _i} item(s) dropped); the "
                f"server takes {LOOK_MAX_CROPS} images per request. "
                "Ask for the rest in another look().")
            break
        f = f.strip()
        if not f:
            continue
        mark_list: list[dict] = []
        for mf in dict.fromkeys([*own, f]):
            xy = resolve(own.get(mf, mf))
            if xy is None:
                skipped.append(mf)
                continue
            mark_list.append(_mark(mf, rows.get(mf), xy, mf == f))
        row = rows.get(f)
        xy = resolve(f)
        if xy is None:
            skipped.append(f)
            continue
        if not _covered(xy):
            uncovered.append(f)
            continue
        requests.append({
            "center_fid": f,
            "center_xy": xy,
            "frame_up": frame[0],
            "frame_note": frame[1],
            "center_label": look_label(f, row, is_target=True),
            "center_boundary": look_outline(row),
            "marks": mark_list,
            # The observation extent: the renderer's window ceiling.
            "obs_span_m": obs_span_m,
        })
    if uncovered:
        _u = ", ".join(uncovered[:6])
        _um = (f", +{len(uncovered) - 6} more"
               if len(uncovered) > 6 else "")
        ref_notes.append(
            f"{len(uncovered)} crop(s) not rendered ({_u}{_um}): that ground "
            f"lies more than {float(obs_minor_m) / 2.0:.0f} m from every "
            "flown stop, outside the middle of every frame so far. A look "
            "shows ground within that reach of a stop"
            + ("." if withheld("library") else
               " -- explore the place so a pass covers it, then look."))
    return requests, skipped, ref_notes


def map_spec(around: str, mark_rows: list, *, rows: dict, resolve,
             frame: tuple) -> tuple[dict | None, list[str]]:
    """One show_map's render spec centred on ``around``, its marks the rows
    ``mark_rows``, each placed by its own token: ``(spec or None,
    skipped_fids)``."""
    cxy = resolve(around)
    if cxy is None:
        return None, [str(around)]
    skipped: list[str] = []
    center_fid = (str(around).strip()
                  if isinstance(around, str)
                  and str(around).strip() in rows else "")
    center_label = (look_label(center_fid, rows.get(center_fid),
                               is_target=False)
                    if center_fid else f"[{cxy[0]}, {cxy[1]}]")
    marks: list[dict] = []
    own = {}
    for r in mark_rows:
        own.setdefault(str(r["fid"]), row_token(r))
    for mf, tok in own.items():
        row = rows.get(mf)
        # a computed row (a crossing, a terminus) is no surface row: its
        # point is what its token names
        xy = look_xy(row) if row is not None else resolve(tok)
        if xy is None:
            skipped.append(mf)
            continue
        marks.append(_mark(mf, row, xy, False))
    # The frame follows the request: the marks' and centre's extent derives
    # the zoom, and a bare point gets the floor. A window that would miss
    # the centre frames the centre alone, so the map is centred there.
    _cb = look_outline(rows.get(center_fid)) if center_fid else None

    def _pts(srcs) -> list:
        return [[float(_b[0]), float(_b[1])] for _src in srcs
                for _b in (_src or [])
                if isinstance(_b, (list, tuple)) and len(_b) >= 2]

    def _frame(pts) -> tuple:
        if not pts:
            return cxy, _MAP_FRAME_MIN_M
        _fx = [p[0] for p in pts]
        _fy = [p[1] for p in pts]
        return ([round((min(_fx) + max(_fx)) / 2.0, 2),
                 round((min(_fy) + max(_fy)) / 2.0, 2)],
                max((max(_fx) - min(_fx)) * _MAP_FRAME_MARGIN,
                    (max(_fy) - min(_fy)) * _MAP_FRAME_MARGIN,
                    _MAP_FRAME_MIN_M))

    _own = _pts([_cb])
    fxy, span = _frame(_pts([[m["xy"] for m in marks if m.get("xy")]]
                            + [m.get("boundary") for m in marks]) + _own)
    if max(abs(cxy[0] - fxy[0]), abs(cxy[1] - fxy[1])) > span / 2.0:
        fxy, span = _frame(_own)
    cxy = fxy
    spec: dict = {
        "center_xy": cxy,
        "context_m": round(float(span), 2),
        "center_fid": center_fid,
        "center_label": center_label,
        "center_boundary": _cb,
        "marks": marks,
    }
    frame_up, frame_note = frame
    if frame_up is not None:
        spec["frame_up"], spec["frame_note"] = frame_up, frame_note
    return spec, skipped


def map_features(ctx) -> list[dict]:
    """The live graph as the map painter's feature dicts: label and paint
    kind, cells, and a road's centerline or a building's contour in world
    metres."""
    from maps.render.osm_palette import (  # noqa: PLC0415
        osm_kind,
        osm_label,
    )
    nodes = ctx.graph.nodes

    feats: list[dict] = []
    for fid, node in nodes.copy().items():
        # The node's own cell array rides as is (no per-map list copy).
        cells = node.cells
        if cells is None or len(cells) == 0:
            c = node.centroid
            cells = ([[int(c[0]), int(c[1])]] if c is not None else [])
        prov = node.provenance
        tags = node.tags
        if prov == "detected":
            label = str(node.class_name or "detection")
            kind = "detection"
        else:
            label = osm_label(
                tags, str(node.class_name or "prior"))
            kind = osm_kind(tags)
        feat = {
            "fid": str(fid), "label": label, "kind": kind,
            "cells": cells,
            "provenance": ("detected" if prov == "detected"
                           else "prebaked"),
        }
        # The name rides beside the class label the palette keys off.
        _nm = tags.get("name")
        if _nm:
            feat["name"] = str(_nm)
        if kind == "road":
            cl = node.centerline
            if cl is not None and len(cl) >= 2:
                feat["centerline"] = [
                    list(cell_centre_xy(r, c, ctx.tile_origin_xy,
                                        ctx.cell_size_m)) for r, c in cl]
        elif kind == "building":
            wc = node.world_contour
            if wc is not None and len(wc) >= 3:
                feat["contour"] = [[float(x), float(y)]
                                   for x, y in wc]
        feats.append(feat)
    return feats


def map_vision_text(maps: list[dict]) -> str:
    """The map message's text: what the picture is (world state, not
    imagery), and each map's window, badge legend and class colors."""
    order = ", ".join(str(m.get("center_fid") or "map") for m in maps)
    _dn = ("Detections are the bright class-colored shapes, each carrying a "
           "small dN chip -- dN is detection fid detected_N, so any "
           "one of them can be named. ")
    if withheld("prior"):
        base = (f"World-state maps -- everything detected so far, not a "
                f"photograph; this run has no prior map, so dark is "
                f"nothing recorded. {_dn}Nothing undetected has a shape "
                f"here; what has been DETECTED is drawn where it "
                f"was found. To see what is actually there, ")
    else:
        base = (f"World-state maps -- the survey plus everything detected "
                f"so far, not a photograph: light-gray lines are roads, "
                f"tan shapes are surveyed buildings, muted regions are "
                f"landuse by class, dark is nothing recorded. {_dn}"
                f"Surveyed buildings big enough "
                f"at this zoom carry their own NNNN chip of bare digits "
                f"(a map around a fid or with marks chips only its centre "
                f"and marks): NNNN is "
                f"prior fid prior_NNNN (narrow the window to chip more of "
                f"them). A detection lying on a surveyed footprint shows "
                f"tan hatching through its fill, and that survey's "
                f"NNNN chip sits stacked under the dN chip -- "
                f"co-location drawn, identity still the graph's call. "
                f"The survey part does NOT have "
                f"the shape of a cul-de-sac bulb, a driveway, a pool, a "
                f"car, or anything built since the survey -- no zoom "
                f"shows those; what has been DETECTED is drawn where it "
                f"was found. To see what is actually there, ")
    base += (f"{'fly: ' if withheld('look') else 'fly and look: '}"
             f"the camera is the live check on this. One per item in "
             f"order: [{order}]. A white outline or ring marks the "
             f"centre; per-map legend:")
    for k, m in enumerate(maps):
        line = " ; ".join(f"{e['label']} ({e['role']})"
                          for e in m.get("legend") or [])
        _span = _window_caption(m)
        base += f"\n  map {k + 1}{_span}: {line}" if line else \
            f"\n  map {k + 1}{_span}:"
        _cls = m.get("classes") or []
        if _cls:
            base += (" ; classes: " + ", ".join(
                f"{c['label']}={c['color']}" for c in _cls))
        elif m.get("features_painted") == 0:
            base += (" ; NOTE: 0 recorded features in this window -- "
                     "the dark image is an empty window, not a broken "
                     "render; re-center the map where rows are.")
        _pu = int(m.get("priors_unchipped") or 0)
        if _pu:
            base += (f" ; {int(m.get('priors_chipped') or 0)} "
                     f"buildings carry fid chips, {_pu} more in frame "
                     f"are unchipped at this zoom -- narrow the window "
                     f"to id them")
        _mof = m.get("marks_out_of_view") or []
        if _mof:
            base += (f" ; requested mark(s) {', '.join(_mof)} lie "
                     f"outside this map's window and are NOT drawn")
    return base
