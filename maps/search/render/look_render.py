"""The ``look`` crop renderer: pure functions of resolved crop specs and the
raster (or the run's own imagery), so a live crop and its re-render are the
same bytes."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# A crop is a digital crop of the captured nadir ortho. Its window fits the
# hull of the request's points (centre, marks, footprints), floored at 3x the
# subject's bbox diagonal and MIN_PX, ceilinged at the stop's view square and
# the pixel budget. Past a ceiling a far mark falls off, never the subject.
LOOK_CTX_MIN_PX: int = 150
# A bare [x, y] subject asks about its surroundings: it gets a window that
# holds them. A maskless fid keeps the tight floor.
LOOK_POINT_CTX_M: float = 60.0
# 1024 visual tokens = a 1024x1024 px image; the serve sets no pixel cap.
LOOK_PIXEL_BUDGET_PX: int = 1024 * 1024
# W x (W + 20 px band) <= BUDGET_PX solves at W = 1014, taken down to 1000.
LOOK_CTX_BUDGET_PX: int = 1000
# One car of margin around the hull. The floor must exceed it so the recenter
# cap (half_ctx - margin/2) stays positive: MIN_PX clears it down to 0.04 m/px.
LOOK_CAR_MARGIN_M: float = 6.0
# The Lanczos upscale (pixels, not detail) aims the side here, factor 1..3.
LOOK_TARGET_PX: int = 450
# Badges per crop; the centre and a target are never dropped.
LOOK_MAX_BADGES: int = 25
# The rows of a set one crop badges (the centre costs one). Every served row
# budget derives from it, so the rows the model reads are the rows it is shown.
LOOK_SET_ROWS: int = LOOK_MAX_BADGES - 1


def look_window_bounds_m(gsd_m: float) -> tuple[float, float]:
    """``(floor_m, ceiling_m)``: the pixel bounds in metres on this grid."""
    g = float(gsd_m)
    if not g > 0:
        raise ValueError(f"ortho gsd must be positive, got {gsd_m!r}")
    return LOOK_CTX_MIN_PX * g, LOOK_CTX_BUDGET_PX * g


def subject_window_containment(
    world_window, subject_xy, boundary=None,
) -> tuple[bool, bool]:
    """``(subject_in_window, centroid_in_window)``: the centroid is in the
    window, else any part of the footprint is (Sutherland-Hodgman clip)."""
    wx0, wy0, wx1, wy1 = (float(v) for v in world_window)
    sx, sy = float(subject_xy[0]), float(subject_xy[1])
    centroid_in = bool(wx0 <= sx <= wx1 and wy0 <= sy <= wy1)
    if centroid_in:
        return True, True
    poly = xy_list(boundary)
    if len(poly) < 3:
        return centroid_in, centroid_in
    for inside, at in (
        (lambda p: p[0] >= wx0, lambda a, b: _isect_x(a, b, wx0)),
        (lambda p: p[0] <= wx1, lambda a, b: _isect_x(a, b, wx1)),
        (lambda p: p[1] >= wy0, lambda a, b: _isect_y(a, b, wy0)),
        (lambda p: p[1] <= wy1, lambda a, b: _isect_y(a, b, wy1)),
    ):
        clipped: list[tuple[float, float]] = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            if inside(cur):
                if not inside(prev):
                    clipped.append(at(prev, cur))
                clipped.append(cur)
            elif inside(prev):
                clipped.append(at(prev, cur))
        poly = clipped
        if len(poly) < 3:
            return False, False
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return bool(abs(area) / 2.0 > 1e-9), False


def resolve_window_center(
    true_xy, ctr_xy, bounds, half_ctx: float, car_margin_m: float,
) -> tuple[float, float]:
    """The window centre: ``ctr_xy`` shifted inside the raster ``bounds``
    (None: unshifted), with its total offset from the subject ``true_xy``
    capped per axis at ``half_ctx - car_margin_m / 2`` so the subject stays
    in frame; past the cap the off-raster part renders black."""
    cx, cy = float(ctr_xy[0]), float(ctr_xy[1])
    tx, ty = float(true_xy[0]), float(true_xy[1])
    if bounds is None:
        return cx, cy
    bx0, by0, bx1, by1 = (float(v) for v in bounds)

    def _clamp(c: float, lo: float, hi: float) -> float:
        # a raster narrower than the window centres it
        return (lo + hi) / 2.0 if hi < lo else max(lo, min(hi, c))

    px = _clamp(cx, bx0 + half_ctx, bx1 - half_ctx)
    py = _clamp(cy, by0 + half_ctx, by1 - half_ctx)
    keep = max(0.0, half_ctx - car_margin_m / 2.0)
    px = min(max(px, tx - keep), tx + keep)
    py = min(max(py, ty - keep), ty + keep)
    return px, py


def _isect_x(a, b, x):
    t = (x - a[0]) / ((b[0] - a[0]) or 1e-12)
    return (x, a[1] + t * (b[1] - a[1]))


def _isect_y(a, b, y):
    t = (y - a[1]) / ((b[1] - a[1]) or 1e-12)
    return (a[0] + t * (b[0] - a[0]), y)


# Bump on any change to the pixels a given spec renders to.
LOOK_RENDERER_VERSION: int = 4


def up_bearing_deg(up_unit) -> float:
    """Compass bearing (0 = N, clockwise) of a world up-unit."""
    import math  # noqa: PLC0415
    return math.degrees(math.atan2(float(up_unit[0]),
                                   float(up_unit[1]))) % 360.0


def frame_up_angle_deg(ux: float, uy: float) -> float:
    """The cv2.getRotationMatrix2D angle (deg) that brings world direction
    ``(ux, uy)`` to image-up on a north-up image: atan2(ux, uy), since the
    screen vector of a world direction is (ux, -uy). East -> +90."""
    import math as _m
    n = _m.hypot(ux, uy) or 1.0
    return _m.degrees(_m.atan2(ux / n, uy / n))


def xy_list(pts) -> list[tuple[float, float]]:
    """The well-formed ``[x, y]`` pairs of ``pts`` as float tuples."""
    return [(float(p[0]), float(p[1])) for p in (pts or [])
            if isinstance(p, (list, tuple)) and len(p) == 2]


def view_rotation(frame_up, w: int, h: int):
    """``(matrix, band_text)`` for a declared view-up on a w x h image: the
    cv2 matrix that turns ``frame_up`` to image-up (None when it is absent
    or north) and the orientation stamp the margin band prints."""
    import cv2  # noqa: PLC0415
    if (isinstance(frame_up, (list, tuple)) and len(frame_up) == 2
            and (float(frame_up[0]) or float(frame_up[1]))):
        a = frame_up_angle_deg(float(frame_up[0]), float(frame_up[1]))
        if abs(a) > 1e-9:
            return (cv2.getRotationMatrix2D((w / 2.0, h / 2.0), a, 1.0),
                    f"up {round(up_bearing_deg(frame_up)) % 360} deg")
    return None, "N up"


def pixel_mapper(x0, y1, sx, sy, w, h, rot):
    """World -> delivered pixel for a north-up window (row 0 = max world y)
    of ``sx`` x ``sy`` metres on a w x h image, then the view rotation."""
    def w2p(wx, wy):
        px = (float(wx) - x0) / sx * w
        py = (y1 - float(wy)) / sy * h
        if rot is not None:
            px, py = (rot[0, 0] * px + rot[0, 1] * py + rot[0, 2],
                      rot[1, 0] * px + rot[1, 1] * py + rot[1, 2])
        return px, py
    return w2p


def frame_arcs(boundary, w2p, w: int, h: int):
    """A world boundary as the OPEN pixel arcs that lie in frame, or None.

    A footprint bigger than its window is clipped, never dropped, and no
    edge runs along the frame edge (that would say the shape ends there).
    Half a stroke of buffer keeps a footprint hugging the frame."""
    from maps.util.polyline import clip_ring_to_box  # noqa: PLC0415
    if not (isinstance(boundary, list) and len(boundary) >= 3):
        return None
    pts = [w2p(*p) for p in xy_list(boundary)]
    if len(pts) < 3:
        return None
    return clip_ring_to_box(pts, (-2.0, -2.0, w + 2.0, h + 2.0)) or None


def frame_echo(spec: dict) -> dict:
    """The record's echo of a declared view (``frame_up`` + ``frame_note``)."""
    fu = spec.get("frame_up")
    if fu is None:
        return {}
    return {"frame_up": [round(float(fu[0]), 4), round(float(fu[1]), 4)],
            "frame_note": str(spec.get("frame_note") or "")}


LOOK_OUTLINE_TOL_M: float = 1.0
LOOK_OUTLINE_MAX_VERTS: int = 256


def look_label(fid: str, row: object, *, is_target: bool) -> str:
    """The legend text for one fid: name or class, the fid itself (what the
    next call has to say), and the detector's confidence and area."""
    name = row.get("name") if isinstance(row, dict) else None
    cls = str(row.get("class") or "") if isinstance(row, dict) else ""
    if name:
        base = f"{str(name)[:24]} {fid}"
    elif cls:
        base = f"{cls} {fid}".strip()
    else:
        base = fid
    bits: list = []
    if isinstance(row, dict):
        _c = row.get("confidence")
        if isinstance(_c, (int, float)) and not isinstance(_c, bool):
            bits.append(f"{float(_c):.2f} conf")
        _a = row.get("footprint_area_m2")
        if isinstance(_a, (int, float)) and not isinstance(_a, bool) and _a:
            bits.append(f"{float(_a):.0f} m2")
    if bits:
        base = f"{base} — {', '.join(bits)}"
    return f"TARGET {base}" if is_target else base


def look_xy(row: object) -> list[float] | None:
    """A row's mark position at the 2 dp the manifest persists."""
    xy = row.get("xy") if isinstance(row, dict) else None
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    return None


def look_outline(row: object) -> list | None:
    """The display outline of a row: the largest loop of its ``mask``,
    Douglas-Peucker at half a 2 m cell (budget 256 vertices), at 2 dp."""
    from maps.util.polyline import simplify_ring  # noqa: PLC0415

    m = row.get("mask") if isinstance(row, dict) else None
    if not (isinstance(m, list) and m):
        return None

    def _area2(lp):
        s = 0.0
        n = len(lp)
        for i in range(n):
            x1, y1 = lp[i]
            x2, y2 = lp[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s)

    loop = simplify_ring(max(m, key=_area2),
                         tolerance=LOOK_OUTLINE_TOL_M,
                         max_points=LOOK_OUTLINE_MAX_VERTS)
    return [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in loop]


def render_candidate_crops(
    requests: list[dict],
    *,
    map_name: str,
    rgbd_dir: str,
    imagery=None,
) -> list[dict]:
    """One crop record per request (``reasoner_loop._build_crop_requests``),
    in order: the window is sized and centred here, then
    ``render_look_from_spec`` renders it. ``imagery`` (``gsd_m`` and
    ``ortho(window)``) is the run's own captures under a live station; unset,
    the raster is the camera. A failed crop is logged and skipped."""
    from maps.perception.ortho.camera_model import (  # noqa: PLC0415
        ortho_gsd_m,
        ortho_world_bounds,
    )

    # The raster extent a near-edge window shifts inside (the run's imagery
    # has no edge). An unreadable ortho renders no crops, so the 0.1 m/px
    # fallback only sizes specs that then fail.
    _wb = None
    if imagery is None:
        try:
            _wb = ortho_world_bounds(map_name, rgbd_dir=rgbd_dir)
        except Exception as exc:  # noqa: BLE001 — no bounds -> just don't shift
            log.debug("look: no raster bounds for %s (%s); no edge-shift",
                      map_name, exc)
    if imagery is not None:
        _gsd = float(imagery.gsd_m)
    else:
        try:
            _gsd = ortho_gsd_m(map_name, rgbd_dir=rgbd_dir)
        except Exception as exc:  # noqa: BLE001
            _gsd = 0.10
            log.debug("look: no ortho gsd for %s (%s); sizing at 0.1 m/px",
                      map_name, exc)
    _min_m, _budget_m = look_window_bounds_m(_gsd)

    out: list[dict] = []
    for req in requests:
        cxy = req.get("center_xy")
        if not (isinstance(cxy, (list, tuple)) and len(cxy) == 2):
            continue
        center_fid = str(req.get("center_fid") or "")
        true_x, true_y = float(cxy[0]), float(cxy[1])
        pts = [(true_x, true_y)]
        for _m in (req.get("marks") or []):
            _mxy = _m.get("xy") if isinstance(_m, dict) else None
            if isinstance(_mxy, (list, tuple)) and len(_mxy) == 2:
                pts.append((float(_mxy[0]), float(_mxy[1])))
            if isinstance(_m, dict):
                pts += xy_list(_m.get("boundary"))
        _xs = [p[0] for p in pts]
        _ys = [p[1] for p in pts]
        _hull = max(max(_xs) - min(_xs), max(_ys) - min(_ys)) if len(pts) > 1 else 0.0
        _diag = 0.0
        _bpts = xy_list(req.get("center_boundary"))
        if len(_bpts) >= 2:
            _bdx = max(p[0] for p in _bpts) - min(p[0] for p in _bpts)
            _bdy = max(p[1] for p in _bpts) - min(p[1] for p in _bpts)
            _diag = (_bdx * _bdx + _bdy * _bdy) ** 0.5
        _is_point = not str(req.get("center_fid") or "").strip()
        _floor = LOOK_POINT_CTX_M if (_is_point and _diag <= 0) else _min_m
        _ctx = max(_floor, 3.0 * _diag, _hull + LOOK_CAR_MARGIN_M)
        _span = req.get("obs_span_m")
        if isinstance(_span, (int, float)) and _span > 0:
            _ctx = min(_ctx, max(_min_m, float(_span)))
        _ctx = min(_ctx, _budget_m)
        half_ctx = _ctx / 2.0
        if len(pts) > 1:
            # the hull centre (the window is sized off the hull, so only its
            # centre is sure to contain it), capped so the subject keeps a
            # half-car margin inside each edge
            ctr_x = (max(_xs) + min(_xs)) / 2.0
            ctr_y = (max(_ys) + min(_ys)) / 2.0
            _keep = max(0.0, half_ctx - LOOK_CAR_MARGIN_M / 2.0)
            ctr_x = min(max(ctr_x, true_x - _keep), true_x + _keep)
            ctr_y = min(max(ctr_y, true_y - _keep), true_y + _keep)
        else:
            ctr_x, ctr_y = true_x, true_y
        pose_x, pose_y = resolve_window_center(
            (true_x, true_y), (ctr_x, ctr_y), _wb, half_ctx, LOOK_CAR_MARGIN_M)
        # built from the rounded values it persists: a byte-exact re-render
        rec = render_look_from_spec({
            "map_name": map_name,
            "center_fid": center_fid,
            "center_label": str(req.get("center_label")
                                or center_fid or "candidate"),
            "center_boundary": req.get("center_boundary"),
            "marks": req.get("marks") or [],
            "world_xy": [round(true_x, 2), round(true_y, 2)],
            "pose_xy": [round(pose_x, 2), round(pose_y, 2)],
            "context_m": _ctx,
            **({"frame_up": req["frame_up"],
                "frame_note": req.get("frame_note") or ""}
               if req.get("frame_up") is not None else {}),
        }, rgbd_dir=rgbd_dir, imagery=imagery)
        if rec is not None:
            out.append(rec)
    return out


def render_look_from_spec(spec: dict, *, rgbd_dir: str, imagery=None) -> dict | None:
    """Render one crop from its spec (``map_name, center_*, marks, world_xy,
    pose_xy, context_m``, optional ``frame_up``): ``{"jpeg": bytes, ...}``,
    whose non-jpeg part is the persisted manifest record and re-renders to the
    same ``sha256``. None (logged) on a malformed spec or a failed render."""
    import hashlib  # noqa: PLC0415
    import cv2  # noqa: PLC0415 — hard dep of the overlay renderer
    import numpy as np  # noqa: PLC0415
    from maps.perception.ortho.camera_model import raster_source_gsd_m  # noqa: PLC0415
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415
    from maps.perception.viz.mllm_overlay import render_look_overlay  # noqa: PLC0415
    from maps.perception.observation import crop_ortho_observation  # noqa: PLC0415

    map_name = str(spec.get("map_name") or "")
    if map_name not in GROUND_LEVEL:
        log.warning("look: map %r not in upstream GROUND_LEVEL; no crop",
                    map_name)
        return None
    ground = float(GROUND_LEVEL[map_name])
    wxy = spec.get("world_xy")
    pxy = spec.get("pose_xy")
    try:
        _ctx = float(spec.get("context_m") or 0.0)
    except (TypeError, ValueError):
        _ctx = 0.0
    if not (isinstance(wxy, (list, tuple)) and len(wxy) == 2
            and isinstance(pxy, (list, tuple)) and len(pxy) == 2
            and _ctx > 0.0):
        log.warning("look: malformed crop spec (world_xy/pose_xy/context_m); "
                    "no crop")
        return None
    true_x, true_y = float(wxy[0]), float(wxy[1])
    pose_x, pose_y = float(pxy[0]), float(pxy[1])
    half_ctx = _ctx / 2.0
    center_fid = str(spec.get("center_fid") or "")
    req = spec
    try:
        _window = (pose_x - half_ctx, pose_y - half_ctx, pose_x + half_ctx, pose_y + half_ctx)
        oc = crop_ortho_observation(
            map_name=map_name,
            pose_xy=(pose_x, pose_y),
            altitude_m=ground + half_ctx,
            yaw_deg=0.0,
            ground_z=ground,
            rgbd_dir=rgbd_dir,
            ortho=imagery.ortho(_window) if imagery is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 — a render fail skips one crop
        log.warning("look: crop render failed at %s: %s", (true_x, true_y),
                    exc)
        return None
    # north-up: row 0 is the window's max world y
    rgb = np.ascontiguousarray(oc["rgb"])
    validity = np.ascontiguousarray(oc["validity"])
    _up = max(1, min(3, round(LOOK_TARGET_PX / max(1, rgb.shape[1]))))
    if _up > 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * _up, rgb.shape[0] * _up),
                         interpolation=cv2.INTER_LANCZOS4)
    H, W = rgb.shape[:2]
    # A declared view (``frame_up``) rotates pixels and ink together; the
    # unrotated validity0 serves the subject sample, the rotated validity the
    # valid fraction.
    # ponytail: whole-image warp (corners go black); a sqrt(2) over-crop
    # upgrade keeps full coverage if framed crops prove load-bearing.
    _rotM, band_text = view_rotation(spec.get("frame_up"), W, H)
    validity0 = validity
    if _rotM is not None:
        rgb = cv2.warpAffine(rgb, _rotM, (W, H))
        _Hv0, _Wv0 = validity.shape[:2]
        validity = cv2.warpAffine(
            validity.astype(np.uint8),
            view_rotation(spec.get("frame_up"), _Wv0, _Hv0)[0], (_Wv0, _Hv0),
            flags=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = oc["world_window"]
    meta = {
        "center_fid": center_fid,
        "center_label": str(req.get("center_label")
                            or center_fid or "candidate"),
        "center_boundary": req.get("center_boundary"),
        "marks": req.get("marks") or [],
        "map_name": map_name,
        "renderer_version": LOOK_RENDERER_VERSION,
        "renderer_env": {"cv2": str(cv2.__version__),
                         "numpy": str(np.__version__)},
        "world_xy": [round(true_x, 2), round(true_y, 2)],
        "pose_xy": [round(pose_x, 2), round(pose_y, 2)],
        "context_m": _ctx,
        "gsd_m_per_px": round((x1 - x0) / max(W, 1), 4),
        "side_px": int(W),
        # what the camera resolved: the live imager's, or the raster's cell
        "source_gsd_m": (round(float(imagery.gsd_m), 4) if imagery is not None
                         else raster_source_gsd_m(str(rgbd_dir), map_name)),
        "source": "run" if imagery is not None else "bake",
        "world_window": [round(float(v), 2) for v in (x0, y0, x1, y1)],
    }
    meta.update(frame_echo(spec))
    sx = (x1 - x0) or 1.0
    sy = (y1 - y0) or 1.0
    _w2p = pixel_mapper(x0, y1, sx, sy, W, H, _rotM)
    center_px = _w2p(true_x, true_y)
    center_label = meta["center_label"]

    marks = req.get("marks") or None
    target_px: tuple[float, float] | None = None
    target_label = ""
    neighbor_px: list[tuple[float, float]] = []
    target_poly_px = None
    neighbor_polys_px: list = []
    # digits as the overlay draws them: 1 centre, 2 target, 3.. the rest;
    # marks past LOOK_MAX_BADGES are dropped and counted
    legend: list[dict] = [{
        "digit": 1, "fid": str(center_fid or ""),
        "label": center_label, "role": "center",
    }]
    _dropped_marks = 0
    _marks_out_of_view: list[str] = []
    _marks_unbadged_in_frame: list[str] = []
    for mk in (marks or []):
        mxy = mk.get("xy") if isinstance(mk, dict) else None
        if not (isinstance(mxy, (list, tuple)) and len(mxy) == 2):
            continue
        px, py = _w2p(mxy[0], mxy[1])
        if not (0.0 <= px <= W and 0.0 <= py <= H):
            # out of view unless part of its footprint is in frame
            _min, _ = subject_window_containment(
                meta["world_window"], mxy, mk.get("boundary"))
            _mof = str(mk.get("fid") or "")
            if _mof and _mof != center_fid:
                if _min:
                    _marks_unbadged_in_frame.append(_mof)
                else:
                    _marks_out_of_view.append(_mof)
            continue
        mfid = str(mk.get("fid") or "")
        if mfid and mfid == center_fid:
            continue
        if bool(mk.get("is_target")) and target_px is None:
            target_px = (px, py)
            target_label = str(mk.get("label") or mfid)
            target_poly_px = frame_arcs(mk.get("boundary"), _w2p, W, H)
            legend.append({"digit": 2, "fid": mfid,
                           "label": target_label, "role": "target"})
        else:
            if len(neighbor_px) >= LOOK_MAX_BADGES - 2:
                _dropped_marks += 1
                continue
            neighbor_px.append((px, py))
            neighbor_polys_px.append(frame_arcs(mk.get("boundary"), _w2p, W, H))
            legend.append({"digit": 2 + len(neighbor_px),
                           "fid": mfid,
                           "label": str(mk.get("label") or mfid),
                           "role": "context"})

    try:
        jp = render_look_overlay(
            rgb,
            center_px=center_px,
            center_poly_px=frame_arcs(req.get("center_boundary"), _w2p, W, H),
            target_px=target_px,
            target_poly_px=target_poly_px,
            neighbor_polys_px=neighbor_polys_px,
            neighbor_px=neighbor_px,
            band_text=band_text,
        )
    except Exception as exc:  # noqa: BLE001
        # a crop whose subject cannot be marked is worse than none
        log.warning("look: identification overlay failed, skipping crop "
                    "at %s: %s", (true_x, true_y), exc)
        return None
    # the caption's facts: is the subject in frame, how big, cut by the
    # frame, and on raster data (validity 0 renders black)
    _sin, _cin = subject_window_containment(
        meta["world_window"], (true_x, true_y),
        req.get("center_boundary"))
    meta["subject_in_window"] = _sin
    meta["subject_centroid_in_window"] = _cin
    _cb = xy_list(req.get("center_boundary"))
    if len(_cb) >= 3:
        _bw = max(p[0] for p in _cb) - min(p[0] for p in _cb)
        _bh = max(p[1] for p in _cb) - min(p[1] for p in _cb)
        meta["subject_extent_m"] = [round(_bw, 1), round(_bh, 1)]
        meta["subject_outline_cut"] = bool(
            min(p[0] for p in _cb) < x0 or max(p[0] for p in _cb) > x1
            or min(p[1] for p in _cb) < y0 or max(p[1] for p in _cb) > y1)
    else:
        meta["subject_extent_m"] = None
        meta["subject_outline_cut"] = False
    _Hv, _Wv = validity0.shape[:2]
    meta["window_valid_frac"] = round(
        float((validity > 0).mean()), 3) if validity.size else 0.0
    _spx = int(min(max((true_x - x0) / sx * _Wv, 0.0), _Wv - 1))
    _spy = int(min(max((y1 - true_y) / sy * _Hv, 0.0), _Hv - 1))
    meta["subject_on_data"] = (
        bool(validity0[_spy, _spx] > 0) if (_cin and validity0.size)
        else None)
    meta["legend"] = legend
    meta["marks_dropped"] = _dropped_marks
    meta["marks_out_of_view"] = _marks_out_of_view
    meta["marks_unbadged_in_frame"] = _marks_unbadged_in_frame
    # the delivered bytes' fingerprint (cv2.imencode is deterministic per build)
    jp = bytes(jp)
    meta["sha256"] = hashlib.sha256(jp).hexdigest()
    return {"jpeg": jp, **meta}
