"""The ``look`` crop renderer (ortho only) — a pure function over
resolved crop specs + the block raster.

``render_candidate_crops`` is the renderer handed to the per-stop round as
``look_renderer``: ``SearchLoop`` delegates here with its per-run tile/raster
identity, and offline replay harnesses call it directly with the episode's
``map_name`` + the local RGBD dir. It consumes only its arguments — no loop
state — so live and replay crops are byte-identical by construction.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ADAPTIVE crop window (meters): a DIGITAL crop of the already-captured
# nadir ortho at its own grid spacing, which selects less of the observation
# and never sees more. The window FITS the hull of the request's own point
# set (its center + every mark's xy and footprint): many ids -> one wide
# badged frame for spatial relations, one id -> a tight crop for appearance.
# FLOOR: ~3x the subject's mask bbox diagonal, never below MIN.
# CEILINGS: (1) the stop's observation extent when the request carries it
# (``obs_span_m``); (2) the pixel budget. Past a ceiling a far anchor falls
# off, never the subject (recenter cap below); the legend still names it.
# The floor and the budget are PIXEL counts, converted per map by
# ``look_window_bounds_m``.
# 150 px = the delivered floor a tight subject gets after the <=3x upscale.
LOOK_CTX_MIN_PX: int = 150
# A bare [x, y] subject (a road end, a crossing) has no bbox to scale from,
# and a point is a question about its surroundings, so it gets a window that
# holds them. A maskless FID keeps the tight floor.
LOOK_POINT_CTX_M: float = 60.0
# THE PIXEL BUDGET: 1024 visual tokens = a 1024x1024 px image. The serve
# sets no pixel cap, so this is the only bound on a crop's size.
LOOK_PIXEL_BUDGET_PX: int = 1024 * 1024
# The delivered side that fits it with the 20 px "N up" band:
# W x (W+20) <= BUDGET_PX solves at W = 1014, taken down to 1000.
LOOK_CTX_BUDGET_PX: int = 1000
# ~one car of breathing room around the hull edge. INVARIANT: the derived
# floor stays > CAR_MARGIN_M so the recenter cap (half_ctx - MARGIN/2) stays
# > 0; MIN_PX 150 clears it on any grid down to 0.04 m/px.
LOOK_CAR_MARGIN_M: float = 6.0
# Lanczos upscale (adds pixels, not detail): the factor aims the delivered
# side at TARGET_PX, clamped to [1, 3], so up*native stays within the budget
# side.
LOOK_TARGET_PX: int = 450
# Badge ceiling per crop. Two slots stay reserved for the center and a
# target, which are never dropped.
LOOK_MAX_BADGES: int = 25
# How many rows of a set ONE crop can carry: the center costs a badge, so a
# set crop badges the first LOOK_SET_ROWS rows of its handle. Every budget
# over a served row list derives from this number, so the rows the model
# READS are the rows vision is SHOWN.
LOOK_SET_ROWS: int = LOOK_MAX_BADGES - 1


def look_window_bounds_m(gsd_m: float) -> tuple[float, float]:
    """``(floor_m, ceiling_m)`` for one map's ortho grid spacing.

    Both bounds are pixel counts (see the constants): the floor is the
    delivered side a tight subject is worth, the ceiling is what the model's
    input budget holds. A coarser grid buys a WIDER window for the same
    pixels, not detail (``source_gsd_m`` in every crop record states that).
    """
    g = float(gsd_m)
    if not g > 0:
        raise ValueError(f"ortho gsd must be positive, got {gsd_m!r}")
    return LOOK_CTX_MIN_PX * g, LOOK_CTX_BUDGET_PX * g


def subject_window_containment(
    world_window, subject_xy, boundary=None,
) -> tuple[bool, bool]:
    """``(subject_in_window, centroid_in_window)`` for one crop subject.

    The subject is an OBJECT, not a point, so containment is
    CENTROID-OR-FOOTPRINT: the centroid in the window, else any part of the
    boundary polygon intersecting it (Sutherland-Hodgman clip).
    ``centroid_in_window`` lets the caption state the partial case."""
    wx0, wy0, wx1, wy1 = (float(v) for v in world_window)
    sx, sy = float(subject_xy[0]), float(subject_xy[1])
    centroid_in = bool(wx0 <= sx <= wx1 and wy0 <= sy <= wy1)
    if centroid_in:
        return True, True
    poly = [(float(p[0]), float(p[1])) for p in (boundary or [])
            if isinstance(p, (list, tuple)) and len(p) == 2]
    if len(poly) < 3:
        return centroid_in, centroid_in
    # Clip the footprint polygon to the window; any surviving area = overlap.
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
    """The crop window's center for ONE requested subject (pure math).

    ``ctr_xy`` is the intended center (the hull recentring already
    applied); ``bounds`` the raster world extent (or None). The center
    shifts in-bounds to trim the black out-of-bounds border, but the TOTAL
    shift away from the subject's own position ``true_xy`` is capped PER AXIS
    at ``half_ctx - car_margin_m/2``: THE SUBJECT OWNS ITS WINDOW. Past the
    cap the off-raster remainder renders as the black no-data border it is
    (the validity facts put that in the caption). Per axis, because that is
    the containment condition for an axis-aligned window."""
    cx, cy = float(ctr_xy[0]), float(ctr_xy[1])
    tx, ty = float(true_xy[0]), float(true_xy[1])
    if bounds is None:
        return cx, cy
    bx0, by0, bx1, by1 = (float(v) for v in bounds)

    def _clamp(c: float, lo: float, hi: float) -> float:
        # hi < lo => raster smaller than the window in this axis: center
        # it (unavoidable minimal border), else clamp the window inside.
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


# Bump on ANY change to the pixels ``render_look_from_spec`` produces FROM A
# GIVEN SPEC: its crop/projection math, the overlay in
# ``mllm_overlay.render_look_overlay``, or the jpeg encode params; not the
# spec-building math in ``render_candidate_crops``. The version rides every
# crop record next to its ``sha256``.
LOOK_RENDERER_VERSION: int = 4


def up_bearing_deg(up_unit) -> float:
    """Compass bearing (0 = N, clockwise) of a world up-unit: the orientation
    stamp for a rotated view (band + caption)."""
    import math  # noqa: PLC0415
    return math.degrees(math.atan2(float(up_unit[0]),
                                   float(up_unit[1]))) % 360.0


def frame_up_angle_deg(ux: float, uy: float) -> float:
    """Rotation (deg, for cv2.getRotationMatrix2D) that brings the world
    direction ``(ux, uy)`` to IMAGE-up in the delivered north-up crop.
    Screen vector of a world direction is (ux, -uy) (x right; image y
    grows SOUTH = -world-y), and the cv2 angle that sends screen (sx, sy)
    to image-up (0, -1) is atan2(sx, -sy), so here atan2(ux, uy); the
    identity case is north (0, 1) -> 0. Sign-locked by test both ways:
    up=east (1, 0) -> +90 (east lands at the top of the image)."""
    import math as _m
    n = _m.hypot(ux, uy) or 1.0
    return _m.degrees(_m.atan2(ux / n, uy / n))


# The outline thinner's tolerance and vertex budget (see ``look_outline``).
LOOK_OUTLINE_TOL_M: float = 1.0
LOOK_OUTLINE_MAX_VERTS: int = 256


def look_label(fid: str, row: object, *, is_target: bool) -> str:
    """The badge legend text for one fid. ``row`` is None for an ``[x, y]``
    token, whose token is then the label.

    THE FID ALWAYS RIDES THE LABEL: the name is what a reader recognises and
    the fid is what the next call has to say."""
    name = row.get("name") if isinstance(row, dict) else None
    cls = str(row.get("class") or "") if isinstance(row, dict) else ""
    if name:
        base = f"{str(name)[:24]} {fid}"
    elif cls:
        base = f"{cls} {fid}".strip()
    else:
        base = fid
    # What the row claims to be, with the detector's own numbers (confidence,
    # footprint area) beside it, never a verdict.
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
    """A row's mark position at 2 dp, the precision the look manifest
    persists, so the rendered mark and the re-render spec are the same
    numbers."""
    xy = row.get("xy") if isinstance(row, dict) else None
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    return None


def look_outline(row: object) -> list | None:
    """DISPLAY-ONLY footprint outline for a crop overlay: the largest-area
    loop of the row's exact ``mask``, thinned shape-aware and rounded to
    2 dp. It feeds no relation math. This is the single producer for the
    rendered overlay, the persisted manifest spec and the re-render, so they
    agree by construction.

    Douglas-Peucker at half a cell, budget 256: the tolerance sits below the
    2 m cell quantization, so the stair-steps the raster knows survive; the
    budget is a backstop."""
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
    max_crops: int = 16,
) -> list[dict]:
    """Render nadir candidate crops for the ``look`` tool.

    ``imagery`` is what THIS RUN has captured, when the pixels are a live
    camera's: an object with ``gsd_m`` and ``ortho(window) -> dict`` (the
    station's ``run_imagery.RunImagery``). Given, every crop is cut from it
    and the bake's raster is never read -- the prior is OSM, the imagery is
    the run's -- and ground the run has not imaged is black with
    ``window_valid_frac`` saying how much, captioned unobserved rather than
    off-raster. Unset (the bench), the bake's raster is the camera.

    For each crop spec (``{center_fid, center_xy, center_label,
    center_boundary, marks, ...}`` built by
    ``reasoner_loop._build_crop_requests``): resolve the ADAPTIVE window
    (fit the hull of the spec's own point set, floored at the subject's
    scale, ceilinged at the observation extent and the pixel budget;
    recenter on the hull centre; edge-shift in-bounds under the
    subject-owns-its-window cap, :func:`resolve_window_center`), then render
    through ``render_look_from_spec``, the same deterministic tail a
    re-render calls.

    Returns one RECORD per crop (rank order), clamped to the 16-image server
    budget: ``{"jpeg": bytes}`` plus the resolved spec
    (``map_name/center_fid/center_label/center_boundary/marks/world_xy/
    intended_xy/pose_xy/context_m``) plus the derived audit fields
    (``sha256``, ``renderer_version``, ``edge_shift_m``, ``alt_m``,
    ``upscale``, ``gsd_m_per_px``, ``world_window``, ``marked``,
    ``identified``, ``outlined``, ``legend``, ``subject_in_window``,
    ``subject_centroid_in_window``, ``subject_on_data``,
    ``window_valid_frac``, ``marks_dropped``, ``marks_out_of_view``,
    ``marks_unbadged_in_frame``). The record minus ``jpeg`` IS the persisted
    manifest crop and the re-render spec; ``world_window`` is the shifted
    window verbatim. A per-crop failure is logged and skipped, never
    raised."""
    from maps.perception.ortho.camera_model import (  # noqa: PLC0415
        ortho_gsd_m,
        ortho_world_bounds,
    )
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415

    if map_name not in GROUND_LEVEL:
        log.warning("look: map %r not in upstream GROUND_LEVEL; no crops",
                    map_name)
        return []

    # Raster world extent, to shift a near-edge window in-bounds (the tile
    # edge renders as a black out-of-bounds border).
    _wb = None
    if imagery is None:
        try:
            _wb = ortho_world_bounds(map_name, rgbd_dir=rgbd_dir)
        except Exception as exc:  # noqa: BLE001 — no bounds -> just don't shift
            log.debug("look: no raster bounds for %s (%s); no edge-shift",
                      map_name, exc)
    # The run's imagery has no raster edge to shift away from. The window
    # bounds in metres come off this map's own grid spacing; an unreadable
    # ortho renders no crops at all, so the 0.1 m/px fallback only sizes
    # specs that then fail.
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
    for req in list(requests)[:max_crops]:
        cxy = req.get("center_xy")
        if not (isinstance(cxy, (list, tuple)) and len(cxy) == 2):
            continue
        center_fid = str(req.get("center_fid") or "")
        true_x, true_y = float(cxy[0]), float(cxy[1])
        # ADAPTIVE window: fit to the hull of everything the request names,
        # its center and every mark's position AND FOOTPRINT.
        pts = [(true_x, true_y)]
        for _m in (req.get("marks") or []):
            _mxy = _m.get("xy") if isinstance(_m, dict) else None
            if isinstance(_mxy, (list, tuple)) and len(_mxy) == 2:
                pts.append((float(_mxy[0]), float(_mxy[1])))
            for _p in ((_m.get("boundary") or []) if isinstance(_m, dict)
                       else []):
                if isinstance(_p, (list, tuple)) and len(_p) == 2:
                    pts.append((float(_p[0]), float(_p[1])))
        _xs = [p[0] for p in pts]
        _ys = [p[1] for p in pts]
        _hull = max(max(_xs) - min(_xs), max(_ys) - min(_ys)) if len(pts) > 1 else 0.0
        # FLOOR scales with the SUBJECT (3x its mask bbox diagonal, never
        # below MIN).
        _diag = 0.0
        _bpts = [(float(p[0]), float(p[1]))
                 for p in (req.get("center_boundary") or [])
                 if isinstance(p, (list, tuple)) and len(p) == 2]
        if len(_bpts) >= 2:
            _bdx = max(p[0] for p in _bpts) - min(p[0] for p in _bpts)
            _bdy = max(p[1] for p in _bpts) - min(p[1] for p in _bpts)
            _diag = (_bdx * _bdx + _bdy * _bdy) ** 0.5
        # A bare [x, y] with no fid behind it takes the point window.
        _is_point = not str(req.get("center_fid") or "").strip()
        _floor = LOOK_POINT_CTX_M if (_is_point and _diag <= 0) else _min_m
        _ctx = max(_floor, 3.0 * _diag, _hull + LOOK_CAR_MARGIN_M)
        # CEILING 1: the stop's observed view square (``obs_span_m``), when
        # the request carries it.
        _span = req.get("obs_span_m")
        if isinstance(_span, (int, float)) and _span > 0:
            _ctx = min(_ctx, max(_min_m, float(_span)))
        # CEILING 2: the pixel budget.
        _ctx = min(_ctx, _budget_m)
        half_ctx = _ctx / 2.0
        # RECENTER on the requested point set; a lone subject keeps its own
        # xy.
        if len(pts) > 1:
            # The HULL CENTRE, never the mean: the window is sized off the
            # hull, so only its centre is guaranteed to contain it.
            ctr_x = (max(_xs) + min(_xs)) / 2.0
            ctr_y = (max(_ys) + min(_ys)) / 2.0
            # THE SUBJECT OWNS ITS WINDOW: cap the shift PER AXIS to keep a
            # half-car of margin inside each edge. It bites only when a
            # ceiling clamped the window below the hull (unclamped,
            # _keep >= _hull/2 makes it a no-op).
            _keep = max(0.0, half_ctx - LOOK_CAR_MARGIN_M / 2.0)
            ctr_x = min(max(ctr_x, true_x - _keep), true_x + _keep)
            ctr_y = min(max(ctr_y, true_y - _keep), true_y + _keep)
        else:
            ctr_x, ctr_y = true_x, true_y
        pose_x, pose_y = resolve_window_center(
            (true_x, true_y), (ctr_x, ctr_y), _wb, half_ctx, LOOK_CAR_MARGIN_M)
        # The spec is built from the SAME rounded values it persists, so the
        # manifest record is a byte-exact re-render input.
        rec = render_look_from_spec({
            "map_name": map_name,
            "center_fid": center_fid,
            "center_label": str(req.get("center_label")
                                or center_fid or "candidate"),
            "center_boundary": req.get("center_boundary"),
            "marks": req.get("marks") or [],
            "world_xy": [round(true_x, 2), round(true_y, 2)],
            "intended_xy": [round(ctr_x, 2), round(ctr_y, 2)],
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
    """Render ONE resolved look crop from its persisted spec (pure).

    ``imagery`` is the run's own captures (``render_candidate_crops``); the
    crop is then cut from ``imagery.ortho(window)`` and the record says
    ``source: "run"`` (a bake re-render says ``source: "bake"``).

    The live round and any re-render both call this function and the
    manifest crop record IS the spec, so the pixels the model saw are
    reproducible from text + the raster, and nothing stores jpeg bytes. The
    record's ``sha256`` fingerprints the delivered bytes; ``renderer_version``
    / ``renderer_env`` say what a differing hash moved.

    Consumes ``{map_name, center_fid, center_label, center_boundary, marks,
    world_xy, intended_xy, pose_xy, context_m}``; extra keys (a full manifest
    record) are ignored and every derived field is recomputed. Returns the
    full crop record ``{"jpeg": bytes, ...}`` or None (logged) on a
    validation/render failure."""
    import hashlib  # noqa: PLC0415
    import cv2  # noqa: PLC0415 — hard dep of the overlay renderer
    import numpy as np  # noqa: PLC0415
    from maps.perception.ortho.camera_model import raster_source_gsd_m  # noqa: PLC0415
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415
    from maps.perception.viz.mllm_overlay import render_look_overlay  # noqa: PLC0415
    from maps.perception.observation import crop_ortho_observation  # noqa: PLC0415
    from maps.util.polyline import clip_ring_to_box  # noqa: PLC0415

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
    ixy = spec.get("intended_xy")
    ctr_x, ctr_y = (
        (float(ixy[0]), float(ixy[1]))
        if isinstance(ixy, (list, tuple)) and len(ixy) == 2
        else (pose_x, pose_y))
    half_ctx = _ctx / 2.0
    center_fid = str(spec.get("center_fid") or "")
    req = spec  # the mark/boundary reads below consume the spec directly
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
    # The native render is already north-up (max world y at row 0); the mark
    # projector below shares the same row mapping.
    rgb = np.ascontiguousarray(oc["rgb"])
    validity = np.ascontiguousarray(oc["validity"])
    # Upscale toward the delivered-side target (LOOK_TARGET_PX).
    _up = max(1, min(3, round(LOOK_TARGET_PX / max(1, rgb.shape[1]))))
    if _up > 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * _up, rgb.shape[0] * _up),
                         interpolation=cv2.INTER_LANCZOS4)
    H, W = rgb.shape[:2]
    # DECLARED-VIEW rotation (``frame_up``, a world [ux, uy] unit): rotate the
    # delivered image so that view's up is IMAGE-up. Pixels and ink rotate
    # together (the projector composes the same matrix); new corners render
    # as the black no-data border. ``validity0`` keeps the unrotated copy
    # for the subject-on-data sample; ``validity`` rotates for the delivered
    # valid fraction.
    # ponytail: whole-image warp (corners go black); a sqrt(2) over-crop
    # upgrade keeps full coverage if framed crops prove load-bearing.
    _fup = spec.get("frame_up")
    _rotM = None
    validity0 = validity
    if (isinstance(_fup, (list, tuple)) and len(_fup) == 2
            and (float(_fup[0]) or float(_fup[1]))):
        _a = frame_up_angle_deg(float(_fup[0]), float(_fup[1]))
        if abs(_a) > 1e-9:
            _rotM = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), _a, 1.0)
            rgb = cv2.warpAffine(rgb, _rotM, (W, H))
            _Hv0, _Wv0 = validity.shape[:2]
            _rotMv = cv2.getRotationMatrix2D((_Wv0 / 2.0, _Hv0 / 2.0),
                                             _a, 1.0)
            validity = cv2.warpAffine(
                validity.astype(np.uint8), _rotMv, (_Wv0, _Hv0),
                flags=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = oc["world_window"]
    # The record = the spec echoed verbatim (the re-render input) + the
    # derived audit geometry.
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
        "intended_xy": [round(ctr_x, 2), round(ctr_y, 2)],
        "pose_xy": [round(pose_x, 2), round(pose_y, 2)],
        # how far the edge-clamp moved the crop off its INTENDED center
        "edge_shift_m": round(float(np.hypot(pose_x - ctr_x,
                                             pose_y - ctr_y)), 2),
        "alt_m": round(ground + half_ctx, 2),
        "context_m": _ctx,
        "upscale": _up,
        "gsd_m_per_px": round((x1 - x0) / max(W, 1), 4),
        # the delivered pixel grid (it varies with the upscale)
        "side_px": int(W),
        # what the CAMERA resolved, against which the line above is only
        # grid spacing over a resample factor: the live imager's, or the
        # configured raster's own cell (camera_model.raster_source_gsd_m)
        "source_gsd_m": (round(float(imagery.gsd_m), 4) if imagery is not None
                         else raster_source_gsd_m(str(rgbd_dir), map_name)),
        # whose pixels: this run's own captures, or the bake's raster (the bench's camera)
        "source": "run" if imagery is not None else "bake",
        "world_window": [round(float(v), 2) for v in (x0, y0, x1, y1)],
        "marked": False,
        "outlined": [],
    }
    if spec.get("frame_up") is not None:
        meta["frame_up"] = [round(float(spec["frame_up"][0]), 4),
                            round(float(spec["frame_up"][1]), 4)]
        meta["frame_note"] = str(spec.get("frame_note") or "")
    # World->pixel in the delivered north-up layout (image row 0 = MAX
    # world-y); marks and image share the same row mapping.
    sx = (x1 - x0) or 1.0
    sy = (y1 - y0) or 1.0

    def _w2p(wx, wy, _x0=x0, _y1=y1, _sx=sx, _sy=sy, _W=W, _H=H):
        px = (float(wx) - _x0) / _sx * _W
        py = (_y1 - float(wy)) / _sy * _H
        if _rotM is not None:
            px, py = (_rotM[0, 0] * px + _rotM[0, 1] * py + _rotM[0, 2],
                      _rotM[1, 0] * px + _rotM[1, 1] * py + _rotM[1, 2])
        return px, py

    # CENTER IDENTIFICATION (every crop): the center fid gets the
    # non-occluding mask-lasso mark (footprint outline + digit badge); its
    # name lives in the tool-text legend.
    center_px = _w2p(true_x, true_y)
    center_label = meta["center_label"]

    # Relational context: a distinct in-frame target + other requested
    # candidates, with digit badges (1=center, 2=target, 3..=context)
    # mirrored by the tool-text legend.
    marks = req.get("marks") or None
    target_px: tuple[float, float] | None = None
    target_label = ""
    neighbor_px: list[tuple[float, float]] = []
    # ONLY the marks whose footprint actually reached the pixels.
    outlined: list[dict] = []

    def _proj_poly(boundary):
        """World boundary verts -> the OPEN crop-px arcs that lie in frame.

        A footprint bigger than its window is CLIPPED, never dropped: the
        in-frame arcs draw and the crossings stay gaps. No edge ever runs
        ALONG the frame edge (``clip_ring_to_box`` cuts edge by edge), since
        that would say the shape ends at the window. Vertex fidelity belongs
        to the producer (``look_outline``).
        """
        if not (isinstance(boundary, list) and len(boundary) >= 3):
            return None
        pts = []
        for p in boundary:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pts.append(_w2p(p[0], p[1]))
        if len(pts) < 3:
            return None
        # Half a stroke of buffer, so a footprint hugging the frame keeps
        # its stroke.
        return clip_ring_to_box(pts, (-2.0, -2.0, W + 2.0, H + 2.0)) or None

    target_poly_px = None
    neighbor_polys_px: list = []
    # LEGEND: digits mirror the overlay's assignment (1 = center,
    # 2 = target, 3.. = neighbors in append order); the tool text renders
    # these lines. Neighbors beyond LOOK_MAX_BADGES are dropped and counted.
    legend: list[dict] = [{
        "digit": 1, "fid": str(center_fid or ""),
        "label": center_label, "role": "center",
    }]
    _dropped_marks = 0
    _marks_out_of_view: list[str] = []
    _marks_unbadged_in_frame: list[str] = []
    _place_labels: list = []
    _seen_lab: set = set()
    _neighbor_digits: list = []
    for mk in (marks or []):
        mxy = mk.get("xy") if isinstance(mk, dict) else None
        if not (isinstance(mxy, (list, tuple)) and len(mxy) == 2):
            continue
        px, py = _w2p(mxy[0], mxy[1])
        if not (0.0 <= px <= W and 0.0 <= py <= H):
            # The mark's CENTROID pixel is outside this crop; it is out of
            # view only when centroid-or-footprint containment fails too,
            # else in frame but unbadged.
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
        # Named places on the photograph: ROADS only (a label on a building
        # would cover the evidence pixels), one label per NAME.
        _lab = str(mk.get("label") or "").strip()
        _named_road = (str(mk.get("kind") or "") == "road" and _lab
                       and _lab != mfid)
        if _named_road and _lab not in _seen_lab:
            _seen_lab.add(_lab)
            _place_labels.append((float(px), float(py), _lab))
        if mfid and mfid == center_fid:
            continue  # the center is already the star of this crop
        if bool(mk.get("is_target")) and target_px is None:
            target_px = (px, py)
            target_label = str(mk.get("label") or mfid)
            target_poly_px = _proj_poly(mk.get("boundary"))
            if target_poly_px:
                outlined.append({"fid": mfid, "label": target_label,
                                 "is_target": True})
            legend.append({"digit": 2, "fid": mfid,
                           "label": target_label, "role": "target"})
        else:
            if len(neighbor_px) >= LOOK_MAX_BADGES - 2:
                _dropped_marks += 1
                continue
            neighbor_px.append((px, py))
            _neighbor_digits.append(None if _named_road else 0)
            neighbor_polys_px.append(_proj_poly(mk.get("boundary")))
            if neighbor_polys_px[-1]:
                outlined.append({"fid": mfid,
                                 "label": str(mk.get("label") or ""),
                                 "is_target": False})
            legend.append({"digit": 2 + len(neighbor_px),
                           "fid": mfid,
                           "label": str(mk.get("label") or mfid),
                           "role": "context"})
    # Resolve the ordinal sequence: named roads take None, everything else
    # takes the next digit, so the legend and the drawn badges count the
    # same way. Neighbours start at 3 (1 = centre, 2 = target), the
    # overlay's neighbor_digit_start default.
    _seq = 2
    _ctx_rows = [_le for _le in legend if _le.get("role") == "context"]
    for _i, _slot in enumerate(_neighbor_digits):
        _le = _ctx_rows[_i] if _i < len(_ctx_rows) else None
        if _slot is None:
            # Identified by NAME: no ordinal drawn or listed.
            if _le is not None:
                _le["digit"] = None
            continue
        _seq += 1
        _neighbor_digits[_i] = _seq
        if _le is not None:
            _le["digit"] = _seq

    try:
        jp = render_look_overlay(
            rgb,
            center_px=center_px,
            center_poly_px=_proj_poly(req.get("center_boundary")),
            target_px=target_px,
            target_poly_px=target_poly_px,
            neighbor_polys_px=neighbor_polys_px,
            neighbor_px=neighbor_px,
            # a rotated crop's band states its TRUE up; "N up" only when
            # the pixels really are north-up.
            band_text=("N up" if _rotM is None
                       else f"up {round(up_bearing_deg(_fup)) % 360} deg"),
            # a named road is identified by its name, so it takes no ordinal
            neighbor_digits=_neighbor_digits,
            place_labels=_place_labels,
        )
    except Exception as exc:  # noqa: BLE001
        # A crop whose subject cannot be identified is worse than no crop,
        # so a failed identification overlay skips it.
        log.warning("look: identification overlay failed, skipping crop "
                    "at %s: %s", (true_x, true_y), exc)
        return None
    # The window that rendered vs the subject that was requested, by
    # centroid-or-footprint containment, so the caption never promises
    # pixels the crop lacks.
    _sin, _cin = subject_window_containment(
        meta["world_window"], (true_x, true_y),
        req.get("center_boundary"))
    meta["subject_in_window"] = _sin
    meta["subject_centroid_in_window"] = _cin
    # HOW BIG the subject is, and whether the frame holds all of it, so the
    # caption can say a subject running past the frame is cut.
    _cb = [(float(p[0]), float(p[1])) for p in (req.get("center_boundary") or [])
           if isinstance(p, (list, tuple)) and len(p) == 2]
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
    # RASTER-DATA facts: part of the window can lie past the raster's data
    # edge (validity 0, rendered black), and a subject whose position is in
    # the window can still have no pixels.
    _Hv, _Wv = validity0.shape[:2]
    meta["window_valid_frac"] = round(
        float((validity > 0).mean()), 3) if validity.size else 0.0
    _spx = int(min(max((true_x - x0) / sx * _Wv, 0.0), _Wv - 1))
    _spy = int(min(max((true_y - y0) / sy * _Hv, 0.0), _Hv - 1))
    meta["subject_on_data"] = (
        bool(validity0[_spy, _spx] > 0) if (_cin and validity0.size)
        else None)
    meta["marked"] = bool(marks)
    meta["identified"] = True
    meta["outlined"] = outlined
    meta["legend"] = legend
    meta["marks_dropped"] = _dropped_marks
    meta["marks_out_of_view"] = _marks_out_of_view
    meta["marks_unbadged_in_frame"] = _marks_unbadged_in_frame
    # The delivered-bytes fingerprint, stored instead of the jpeg.
    # cv2.imencode output is deterministic for one cv2 build.
    jp = bytes(jp)
    meta["sha256"] = hashlib.sha256(jp).hexdigest()
    return {"jpeg": jp, **meta}
