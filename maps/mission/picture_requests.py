"""What a look and a map put in front of the model: the render requests they
resolve off the surface rows and the captions that ride their pictures."""

from __future__ import annotations

import math

from maps.library.op_compiler import HANDLE_RE, row_token
from maps.query.spatial_math import cell_centre_xy
from maps.render.look_render import (
    LOOK_MAX_BADGES,
    look_label,
    look_outline,
    look_xy,
    up_bearing_deg,
)
from maps.render.osm_palette import osm_kind, osm_label
from maps.xy_token import parse_xy_token

# The server's per-request image cap.
LOOK_MAX_CROPS = 16

# A framed map's slack around what was asked about, and its tightest frame
# (below it the renderer's GSD floor binds and a tighter frame only removes
# context).
_MAP_FRAME_MARGIN = 1.6
_MAP_FRAME_MIN_M = 200.0


def _legend_line(entries: list[dict]) -> str:
    return " ; ".join(f"{e['digit']} = {e['label']} ({e['role']})" for e in entries)


def _window_caption(c: dict) -> str:
    """One rendered image's window: span, orientation and world extent."""
    # A rotated image states its true up; only an unrotated one says north-up.
    frame_up = c.get("frame_up")
    if isinstance(frame_up, (list, tuple)) and len(frame_up) == 2:
        frame_note = str(c.get("frame_note") or "").strip()
        bearing = round(up_bearing_deg(frame_up)) % 360
        orientation = (
            f"up {bearing} deg"
            + (f" -- {frame_note}" if frame_note else "")
            # a turn off the square leaves the frame's corners empty
            + (
                "; its corners lie outside the turned window and render dark"
                if bearing % 90
                else ""
            )
        )
    else:
        orientation = "north-up"
    context_m = c.get("context_m")
    caption = (
        f" ({context_m:g} m window; {orientation})"
        if isinstance(context_m, (int, float))
        else f" ({orientation})"
    )
    # The numeric extent makes a pixel position back-projectable.
    window = c.get("world_window")
    if isinstance(window, (list, tuple)) and len(window) == 4:
        caption += f" [x {window[0]:g}..{window[2]:g}, y {window[1]:g}..{window[3]:g}]"
    return caption


def look_vision_text(crops: list[dict]) -> str:
    """The look message's text: the crops' detail and order, and each crop's
    window, badge legend (the digit -> fid binding; colour is left to the
    pixels) and what of its subject and marks the image does not show."""

    def _subject(c):
        return str(
            c.get("center_fid")
            or next(
                (
                    e["label"]
                    for e in c.get("legend") or []
                    if e.get("role") == "center" and e.get("label")
                ),
                "?",
            )
        )

    order = ", ".join(_subject(c) for c in crops)
    # GSD from the crop records themselves, never a literal.
    gsds = {round(float(c["gsd_m_per_px"]), 2) for c in crops if c.get("gsd_m_per_px")}
    gsd = (
        f"{min(gsds):.2f} m/px"
        if len(gsds) == 1
        else f"{min(gsds):.2f}-{max(gsds):.2f} m/px"
        if gsds
        else "m/px unknown"
    )
    # Resampling adds pixels, never detail: the source imagery's GSD bounds
    # what the caption may claim.
    source_gsd = max(
        (float(c["source_gsd_m"]) for c in crops if c.get("source_gsd_m")), default=0.0
    )
    detail = (
        f"{gsd}; detail limited by {source_gsd:.2f} m source imagery"
        if source_gsd and source_gsd > min(gsds or {source_gsd})
        else gsd
    )
    text = (
        f"Overhead crops ({detail}, nadir), one per fid "
        f"in order: [{order}]. White outline marks, numbered badges; "
        f"per-crop badge legend. Altitude does NOT change crop detail: "
        f"a crop is a digital crop of imagery already captured, so "
        f"flying lower buys none. What it buys is a NEW view square "
        f"somewhere else:"
    )
    for k, c in enumerate(crops):
        entries = c.get("legend") or []
        if not entries:
            continue
        line = _legend_line(entries)
        window = _window_caption(c)
        # The pixel grid, per image: with the window it converts what the
        # model sees into a world point.
        side_px = c.get("side_px")
        if side_px:
            window += f" ({int(side_px)}x{int(side_px)} px)"
        text += f"\n  crop {k + 1}{window}: {line}"
        subject_xy = c.get("world_xy") or ["?", "?"]
        if c.get("subject_in_window") is False:
            bounds = c.get("world_window") or ["?"] * 4
            text += (
                " ; WARNING: neither this fid's position "
                f"[{subject_xy[0]}, {subject_xy[1]}] nor any part of its "
                f"footprint is inside this crop's window "
                f"[x {bounds[0]}..{bounds[2]}, y {bounds[1]}..{bounds[3]}] "
                "(edge-clamped or off-tile) -- the subject is not "
                "in this image; do not read color or arrangement "
                "from it (the position and window bounds here say "
                "where it sits relative to this crop)."
            )
        elif c.get("subject_centroid_in_window") is False:
            text += (
                " ; note: this fid's recorded position "
                f"[{subject_xy[0]}, {subject_xy[1]}] is outside this crop's "
                "window (edge-clamped at the tile boundary), but "
                "part of its footprint IS in frame -- the outline/"
                "circle marks the in-frame part."
            )
        elif c.get("subject_on_data") is False and c.get("source") == "run":
            text += (
                " ; WARNING: this fid's position "
                f"[{subject_xy[0]}, {subject_xy[1]}] is inside this crop's "
                "window but this run has not imaged that ground yet "
                "-- the black region is unobserved, not a dark "
                "object; its color/shape cannot be read until a leg "
                "flies over it (go_to it, then look)."
            )
        elif c.get("subject_on_data") is False:
            text += (
                " ; WARNING: this fid's position "
                f"[{subject_xy[0]}, {subject_xy[1]}] is inside this crop's "
                "window but past the map raster's DATA edge -- "
                "the black region is off-raster no-data, not a "
                "dark object; the raster holds no pixels for "
                "this subject, so its color/shape cannot be "
                "read from this image."
            )
        # A subject bigger than the frame: state its size, the ink is cut.
        extent = c.get("subject_extent_m")
        if c.get("subject_outline_cut") and isinstance(extent, (list, tuple)):
            text += (
                f" ; note: this fid's footprint is {extent[0]:g} x "
                f"{extent[1]:g} m and runs past this window, so the "
                "outline marks only where its edge crosses the frame "
                "-- not a shape you can read whole here. Its size is "
                "the fact; the ink is a fragment."
            )
        valid_frac = c.get("window_valid_frac")
        if (
            isinstance(valid_frac, (int, float))
            and valid_frac < 0.99
            and c.get("subject_on_data") is not False
        ):
            text += f" ; note: {round((1 - valid_frac) * 100)}% of this " + (
                "window is ground this run has not imaged yet "
                "(black there = unobserved, not shadow)."
                if c.get("source") == "run"
                else "window lies past the map raster's data edge "
                "(black there = no data, not shadow)."
            )
        if int(c.get("marks_dropped") or 0):
            text += (
                f" ; +{int(c['marks_dropped'])} requested mark(s) "
                f"not drawn (badge cap {LOOK_MAX_BADGES}/crop -- "
                f"request them in another look())"
            )
        out_of_view = c.get("marks_out_of_view") or []
        if out_of_view:
            text += (
                f" ; requested mark(s) {', '.join(out_of_view)} lie "
                f"outside this crop's frame and are NOT drawn "
                f"(absent from the image, not unlabeled)"
            )
        unbadged = c.get("marks_unbadged_in_frame") or []
        if unbadged:
            text += (
                f" ; requested mark(s) {', '.join(unbadged)}: part of "
                f"the footprint IS in frame but the centroid badge "
                f"position falls outside it, so no badge is drawn "
                f"(in the image, unlabeled)"
            )
    return text


def rounded_xy(p) -> list[float] | None:
    """An ``[x, y]`` token or point -> the point at centimetre precision,
    else None."""
    if isinstance(p, str):
        p = parse_xy_token(p)
    return [round(float(p[0]), 2), round(float(p[1]), 2)] if p else None


def _mark(fid: str, row, xy, is_target: bool) -> dict:
    """One drawn mark: the row's point, outline and badge label."""
    return {
        "fid": fid,
        "xy": xy,
        "boundary": look_outline(row),
        "label": look_label(fid, row, is_target=is_target),
        "is_target": is_target,
    }


def crop_requests(
    fids: list[str],
    marks: list | None,
    *,
    rows: dict,
    resolve,
    carry,
    stops: dict,
    obs_minor_m: float | None,
    obs_span_m: float | None,
    frame: tuple,
) -> tuple[list[dict], list[str], list[str]]:
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

    def covered(point) -> bool:
        if not (isinstance(obs_minor_m, (int, float)) and obs_minor_m > 0):
            return True
        reach = float(obs_minor_m) / 2.0
        return any(math.dist(point, stop) <= reach for stop in stops.values())

    # A handle is a legal mark (a street is its ways), each row placed by
    # its token_of token (a terminus by its xy, its end_N being per call).
    token_of: dict = {}
    for mark in marks or []:
        if HANDLE_RE.fullmatch(mark):
            for handle_row in carry.read(mark) or []:
                if handle_row.get("fid"):
                    token_of.setdefault(str(handle_row["fid"]), row_token(handle_row))
        else:
            token_of.setdefault(mark, mark)
    for i, fid in enumerate(fids):
        if len(requests) >= LOOK_MAX_CROPS:
            ref_notes.append(
                f"only the first {LOOK_MAX_CROPS} crops of this call "
                f"rendered ({len(fids) - i} item(s) dropped); the "
                f"server takes {LOOK_MAX_CROPS} images per request. "
                "Ask for the rest in another look()."
            )
            break
        fid = fid.strip()
        if not fid:
            continue
        mark_list: list[dict] = []
        for mark_fid in dict.fromkeys([*token_of, fid]):
            xy = resolve(token_of.get(mark_fid, mark_fid))
            if xy is None:
                skipped.append(mark_fid)
                continue
            mark_list.append(_mark(mark_fid, rows.get(mark_fid), xy, mark_fid == fid))
        row = rows.get(fid)
        xy = resolve(fid)
        if xy is None:
            skipped.append(fid)
            continue
        if not covered(xy):
            uncovered.append(fid)
            continue
        requests.append(
            {
                "center_fid": fid,
                "center_xy": xy,
                "frame_up": frame[0],
                "frame_note": frame[1],
                "center_label": look_label(fid, row, is_target=True),
                "center_boundary": look_outline(row),
                "marks": mark_list,
                # The observation extent: the renderer's window ceiling.
                "obs_span_m": obs_span_m,
            }
        )
    if uncovered:
        named = ", ".join(uncovered[:6])
        more = f", +{len(uncovered) - 6} more" if len(uncovered) > 6 else ""
        ref_notes.append(
            f"{len(uncovered)} crop(s) not rendered ({named}{more}): that ground "
            f"lies more than {float(obs_minor_m) / 2.0:.0f} m from every "
            "flown stop, outside the middle of every frame so far. A look "
            "shows ground within that reach of a stop"
            " -- explore the place so a pass covers it, then look."
        )
    return requests, skipped, ref_notes


def map_spec(
    around: str, mark_rows: list, *, rows: dict, resolve, frame: tuple
) -> tuple[dict | None, list[str]]:
    """One show_map's render spec centred on ``around``, its marks the rows
    ``mark_rows``, each placed by its token_of token: ``(spec or None,
    skipped_fids)``."""
    cxy = resolve(around)
    if cxy is None:
        return None, [str(around)]
    skipped: list[str] = []
    center_fid = (
        str(around).strip() if isinstance(around, str) and str(around).strip() in rows else ""
    )
    center_label = (
        look_label(center_fid, rows.get(center_fid), is_target=False)
        if center_fid
        else f"[{cxy[0]}, {cxy[1]}]"
    )
    marks: list[dict] = []
    token_of = {}
    for r in mark_rows:
        token_of.setdefault(str(r["fid"]), row_token(r))
    for fid, tok in token_of.items():
        row = rows.get(fid)
        # a computed row (a crossing, a terminus) is no surface row: its
        # point is what its token names
        xy = look_xy(row) if row is not None else resolve(tok)
        if xy is None:
            skipped.append(fid)
            continue
        marks.append(_mark(fid, row, xy, False))
    # The frame follows the request: the marks' and centre's extent derives
    # the zoom, and a bare point gets the floor. A window that would miss
    # the centre frames the centre alone, so the map is centred there.
    center_boundary = look_outline(rows.get(center_fid)) if center_fid else None

    def points_of(srcs) -> list:
        return [
            [float(point[0]), float(point[1])]
            for source in srcs
            for point in (source or [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]

    def frame_of(pts) -> tuple:
        if not pts:
            return cxy, _MAP_FRAME_MIN_M
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (
            [round((min(xs) + max(xs)) / 2.0, 2), round((min(ys) + max(ys)) / 2.0, 2)],
            max(
                (max(xs) - min(xs)) * _MAP_FRAME_MARGIN,
                (max(ys) - min(ys)) * _MAP_FRAME_MARGIN,
                _MAP_FRAME_MIN_M,
            ),
        )

    center_points = points_of([center_boundary])
    fxy, span = frame_of(
        points_of([[m["xy"] for m in marks if m.get("xy")]] + [m.get("boundary") for m in marks])
        + center_points
    )
    if max(abs(cxy[0] - fxy[0]), abs(cxy[1] - fxy[1])) > span / 2.0:
        fxy, span = frame_of(center_points)
    cxy = fxy
    spec: dict = {
        "center_xy": cxy,
        "context_m": round(float(span), 2),
        "center_fid": center_fid,
        "center_label": center_label,
        "center_boundary": center_boundary,
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
    nodes = ctx.graph.nodes

    feats: list[dict] = []
    for fid, node in nodes.copy().items():
        # The node's own cell array rides as is (no per-map list copy).
        cells = node.cells
        if cells is None or len(cells) == 0:
            c = node.centroid
            cells = [[int(c[0]), int(c[1])]] if c is not None else []
        prov = node.provenance
        tags = node.tags
        if prov == "detected":
            label = str(node.class_name or "detection")
            kind = "detection"
        else:
            label = osm_label(tags, str(node.class_name or "prior"))
            kind = osm_kind(tags)
        feat = {
            "fid": str(fid),
            "label": label,
            "kind": kind,
            "cells": cells,
            "provenance": ("detected" if prov == "detected" else "prebaked"),
        }
        # The name rides beside the class label the palette keys off.
        name = tags.get("name")
        if name:
            feat["name"] = str(name)
        if kind == "road":
            cl = node.centerline
            if cl is not None and len(cl) >= 2:
                feat["centerline"] = [
                    list(cell_centre_xy(r, c, ctx.tile_origin_xy, ctx.cell_size_m)) for r, c in cl
                ]
        elif kind == "building":
            wc = node.world_contour
            if wc is not None and len(wc) >= 3:
                feat["contour"] = [[float(x), float(y)] for x, y in wc]
        feats.append(feat)
    return feats


def map_vision_text(maps: list[dict]) -> str:
    """The map message's text: what the picture is (world state, not
    imagery), and each map's window, badge legend and class colors."""
    order = ", ".join(str(m.get("center_fid") or "map") for m in maps)
    detections = (
        "Detections are the bright class-colored shapes, each carrying a "
        "small dN chip -- dN is detection fid detected_N, so any "
        "one of them can be named. "
    )
    text = (
        f"World-state maps -- the survey plus everything detected "
        f"so far, not a photograph: light-gray lines are roads, "
        f"tan shapes are surveyed buildings, muted regions are "
        f"landuse by class, dark is nothing recorded. {detections}"
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
        f"was found. To see what is actually there, "
    )
    text += (
        "fly and look: "
        "the camera is the live check on this. One per item in "
        f"order: [{order}]. A white outline or ring marks the "
        f"centre; per-map legend:"
    )
    for k, m in enumerate(maps):
        line = " ; ".join(f"{e['label']} ({e['role']})" for e in m.get("legend") or [])
        window = _window_caption(m)
        text += f"\n  map {k + 1}{window}: {line}" if line else f"\n  map {k + 1}{window}:"
        classes = m.get("classes") or []
        if classes:
            text += " ; classes: " + ", ".join(f"{c['label']}={c['color']}" for c in classes)
        elif m.get("features_painted") == 0:
            text += (
                " ; NOTE: 0 recorded features in this window -- "
                "the dark image is an empty window, not a broken "
                "render; re-center the map where rows are."
            )
        unchipped = int(m.get("priors_unchipped") or 0)
        if unchipped:
            text += (
                f" ; {int(m.get('priors_chipped') or 0)} "
                f"buildings carry fid chips, {unchipped} more in frame "
                f"are unchipped at this zoom -- narrow the window "
                f"to id them"
            )
        out_of_view = m.get("marks_out_of_view") or []
        if out_of_view:
            text += (
                f" ; requested mark(s) {', '.join(out_of_view)} lie "
                f"outside this map's window and are NOT drawn"
            )
    return text
