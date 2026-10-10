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

# ADAPTIVE crop window (meters) — a DIGITAL crop of the already-captured
# nadir ortho at its own grid spacing: cropping selects less of the current
# observation, it never sees more. The window FITS the hull of the request's
# own point set (its center + every mark xy), so the agent picks its
# workflow by what it passes: many ids -> one wide badged frame for spatial
# relations, one id -> a tight crop for appearance. Measured basis
# (SoM-ordinal probe, 2026-07, qwen36-35b-a3b): ordering over a badged set
# is flat to 25 badges and to 0.039% object area, while single-object
# identity is what degrades with badge density — the wide frame answers
# relations, the tight crop answers color/kind, and one call carries both
# because crops is a list.
# FLOOR: ~3x the subject's mask bbox diagonal (the subject fills roughly a
# third of its window — enough surround to place it), never below MIN; a
# maskless centroid-only subject keeps the floor window.
# CEILINGS: (1) the stop's observation extent, when the request carries it
# (``obs_span_m`` = the view-square side ``2*(altitude - ground)``) — a
# window can never exceed what the stop observed; (2) the pixel budget
# ``LOOK_PIXEL_BUDGET_PX`` below, which ``LOOK_CTX_BUDGET_PX`` is derived
# against: the overlay adds a fixed 20 px "N up" band, so the budget side
# keeps the delivered W x (W+20) image inside the budget for every span.
# Past a ceiling a far anchor falls off, never the subject (recenter cap
# below); the tool-text legend still names it.
# (The objects.json "contour" is an inflated ~9.5 m annotation blob, NOT the
# basis for any occupancy calc here.)
# The floor and the ceiling are both PIXEL counts, so both are declared in
# pixels and converted per map by ``look_window_bounds_m`` below. Writing
# them as metres hardcoded one map's grid spacing (0.1 m/px) into every
# other map's window: Cal City's ortho is 0.3 m/px, so its 100 m crops
# shipped 333 x 353 px — 11% of the pixel budget — and its 15 m crops
# shipped 50 native px Lanczos-blown to 150. Measured on calcity_final:
# 52.8% of look crops sat exactly at the metre ceiling and 20% carried a
# subject whose 3x diagonal overflowed it (median diagonal 77.6 m, p90
# 382 m), which is the large-anchor case the window exists to serve.
# 150 px = the delivered floor a tight subject gets after the <=3x upscale.
LOOK_CTX_MIN_PX: int = 150
# ...but a MASKLESS subject (a bare [x, y]: a road end, a crossing, a
# cul-de-sac bulb) has no bbox to scale from, so `3 x diag` is zero and it
# falls to the MIN floor — measured at 98-100% of point crops across two
# runs. That floor was tuned for 0.1 m/px SensatUrban; on calcity's 0.6 m
# optical NAIP a 15 m window is ~25x25 real samples, and the shapes asked
# for at a point (a 30 m bulb) overflow it twice over. "Blurry beige" was
# the physically correct percept, and the model said so in 54% of episodes.
# A point is a QUESTION about its surroundings, so it gets a window that
# holds them. Scoped to bare [x, y] subjects: a maskless FID is still a
# small named thing and keeps the tight frame.
LOOK_POINT_CTX_M: float = 60.0
# THE CAMERA'S OWN RESOLUTION, per source map — the limit the delivered
# grid spacing says nothing about. Cal City is NAIP at 0.6 m native
# (configs/calcity.yaml states it), baked to a 0.3 m/px grid and then
# Lanczos-upscaled up to 3x, which is how a crop came to be captioned
# "0.10 m/px". Unknown maps report nothing rather than a guess.
_SOURCE_GSD_M: dict = {"calcity": 0.6}


def _source_gsd_m(map_name: str) -> float | None:
    return _SOURCE_GSD_M.get(str(map_name or "").strip().lower())
# THE PIXEL BUDGET, declared here because here is where it is enforced.
# 1024 visual tokens = a 1024x1024 px image, and ``LOOK_CTX_BUDGET_PX``
# below is the delivered side that keeps the W x (W+20) crop (the 20 px
# band is the "N up" overlay) inside it.
# It is a MODEL input-size fact, not a serve flag: the sglang serve path
# sets no pixel cap of its own, so nothing downstream will clamp an
# oversized crop for us — this constant is the only thing standing between
# a wide span and an image the model reads badly.
LOOK_PIXEL_BUDGET_PX: int = 1024 * 1024
# The delivered side that fits it: W x (W+20) <= BUDGET_PX solves at
# W = 1014, taken down to a round 1000.
LOOK_CTX_BUDGET_PX: int = 1000
# ~one car of breathing room around the hull edge. INVARIANT: keep the
# derived floor > CAR_MARGIN_M so the recenter cap (half_ctx - MARGIN/2)
# stays > 0 and the target can never be pushed out of frame; a floor below
# MARGIN would silently disable that cap. MIN_PX 150 clears it on any grid
# down to 0.04 m/px.
LOOK_CAR_MARGIN_M: float = 6.0
# Lanczos upscale (adds pixels/tokens, NOT detail beyond the source GSD):
# the factor is derived per crop from the native side, aiming the delivered
# side at TARGET_PX and clamped to [1, 3] — a lone 15 m subject ships 450 px
# (x3), a 60 m relational frame ~600 px (x1) — so up*native never exceeds
# the 1024 px budget side.
LOOK_TARGET_PX: int = 450
# Badge ceiling per crop: ordering is measured FLAT to 25 badges (the SoM
# probe's density sweep) and identity is not, so the cap binds only the
# relational frame (a lone-subject crop has no marks). Two slots stay
# reserved for the center and a target, which are never dropped.
LOOK_MAX_BADGES: int = 25
# THE vision budget: how many rows of a set ONE crop can carry. The center is
# a mark that renders as the subject, so it costs a badge and not a neighbour
# slot -- a set crop badges the first LOOK_SET_ROWS rows of its handle. Every
# budget over a served row list derives from this one number (the set-crop
# slice, the broad-result render head, the offline containment cut), so the
# rows the model READS are the rows vision is SHOWN. They were 12 and 24:
# containment@12 is 69.2% where containment@24 is 79.1% on the same ordering
# (containment_probe, 325-episode facing family), so the narrower head was
# throwing away 9.9 points of a budget the renderer had already paid for.
LOOK_SET_ROWS: int = LOOK_MAX_BADGES - 1


def look_window_bounds_m(gsd_m: float) -> tuple[float, float]:
    """``(floor_m, ceiling_m)`` for one map's ortho grid spacing.

    Both bounds are pixel counts (see the constants): the floor is the
    delivered side a tight subject is worth, the ceiling is what the model's
    input budget holds. A coarser grid buys a WIDER window for the same
    pixels — it does not buy detail, which the source imagery caps
    (``source_gsd_m`` in every crop record says so). CityNav's 0.1 m/px grid
    reproduces the historical 15 m / 100 m exactly.
    """
    g = float(gsd_m)
    if not g > 0:
        raise ValueError(f"ortho gsd must be positive, got {gsd_m!r}")
    return LOOK_CTX_MIN_PX * g, LOOK_CTX_BUDGET_PX * g


def subject_window_containment(
    world_window, subject_xy, boundary=None,
) -> tuple[bool, bool]:
    """``(subject_in_window, centroid_in_window)`` for one crop subject.

    The subject is an OBJECT, not a point: an elongated area prior whose
    recorded centroid falls outside an edge-clamped window can still have a
    large part of its FOOTPRINT inside the delivered pixels (measured, 8267:
    a 76.4 m grass strip's centroid sat 10.9 m west of the tile raster while
    the clamped window contained the strip's cue-relevant end — the old
    centroid-point test served a false 'the subject is not in this image').
    So containment is CENTROID-OR-FOOTPRINT: the centroid point in the
    window, else any part of the boundary polygon intersecting it
    (Sutherland–Hodgman clip, pure stdlib). ``centroid_in_window`` rides
    along so the caption can state the truthful partial case ('centroid
    outside; part of the footprint is in frame')."""
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
    shift away from the subject's own position ``true_xy`` is capped
    PER AXIS at ``half_ctx - car_margin_m/2`` — THE SUBJECT OWNS ITS
    WINDOW. Measured (7875, killed 3/5 finalists): three finalists past the
    raster corner were clamped fully in-bounds, two DIFFERENT fids got the
    byte-identical corner window, and none of the three windows contained
    its own subject while the legend read as each fid's neighborhood.
    Past the cap the window stays subject-centered and the off-raster
    remainder renders as the black NO-DATA border it truly is (the
    validity facts put that in the caption).

    PER AXIS, because that IS the containment condition for an axis-aligned
    window: a Euclidean radius cap is the same rule shrunk by up to sqrt(2)
    on a diagonal shift, and it clipped requested rows out of a window wide
    enough to hold them whenever the subject sat at a hull corner (a 24-row
    set spanning 24 m in a 30 m window drew 16 badges, 8 named out-of-view,
    with the window still fully inside the raster)."""
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
# GIVEN SPEC: the crop/projection math inside it, the overlay in
# ``mllm_overlay.render_look_overlay`` (marks, badges, the "N up" band), or
# the jpeg encode params. NOT the spec-BUILDING math in
# ``render_candidate_crops`` / ``resolve_window_center``: that chooses which
# window to ask for, and a record persisted before it moved still re-renders
# byte-identical from its own stored ``pose_xy``/``context_m``. The version
# rides every persisted crop record next to its ``sha256``, so a dashboard
# re-render can say WHY bytes differ (renderer moved) instead of guessing.
# v2: every circle/proxy-shape heuristic deleted — the mark is the object's
# actual mask polygon, or badge-only when the object carries none.
# v3: crops deliver in the native north-up orientation (row 0 = max world
# y = north); the former vertical flip at the LOOK boundary is gone.
LOOK_RENDERER_VERSION: int = 4


def up_bearing_deg(up_unit) -> float:
    """Compass bearing (0 = N, clockwise) of a world up-unit — the truthful
    orientation stamp for a rotated view (band + caption)."""
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


# The outline thinner's two knobs (see ``look_outline``). The tolerance sits
# BELOW the 2 m cell quantization every prior footprint arrives at, so the
# stair-steps survive; the budget is a backstop that never binds on Cal
# City's largest footprints.
LOOK_OUTLINE_TOL_M: float = 1.0
LOOK_OUTLINE_MAX_VERTS: int = 256


def look_label(fid: str, row: object, *, is_target: bool) -> str:
    """The badge legend text for one fid. ``row`` is None for an ``[x, y]``
    token — a mark/center on a bare world point has no surface row, so the
    token itself is the label.

    THE FID ALWAYS RIDES THE LABEL. A named prior printed its name and
    nothing else, so 2,115 of 6,256 served legend entries (33.8%) carried a
    badge whose fid was nowhere on the message -- while the crop's own
    question says "Answer with the badge digit and its fid". The model was
    asked for something the picture never told it. The name is what a reader
    recognises and the fid is what the next call has to say, so a legend that
    means to be acted on carries both."""
    name = row.get("name") if isinstance(row, dict) else None
    cls = str(row.get("class") or "") if isinstance(row, dict) else ""
    if name:
        base = f"{str(name)[:24]} {fid}"
    elif cls:
        base = f"{cls} {fid}".strip()
    else:
        base = fid
    # WHAT THE ROW CLAIMS TO BE, beside the picture of it. A badge carried
    # the class as a bare word, so a crop reading against its label was a
    # coin toss with no odds on it: calcity::200#198 looked at a crop
    # captioned "TARGET house detected_185", wrote "the caption says house,
    # but the image is clearly a red car -- let's trust the visual
    # evidence", and flew there. detected_185 is a red-ROOFED house, class
    # confidence 0.414, 96 m2. At 0.6 m imagery a car is 8-37 source
    # samples and a red roof and a red car are not reliably separable, so
    # the pixels alone cannot settle it -- but 0.41 and 96 m2 can, because
    # a car on that same run's table runs 2.7-13.4 m2.
    #
    # The detector's own numbers, never a verdict: the model still decides.
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
    """A row's mark position at 2 dp (1 cm = 0.1 native px) — the precision
    the look manifest persists; rounding HERE means the rendered mark and the
    recorded re-render spec are the same numbers."""
    xy = row.get("xy") if isinstance(row, dict) else None
    if isinstance(xy, (list, tuple)) and len(xy) == 2:
        return [round(float(xy[0]), 2), round(float(xy[1]), 2)]
    return None


def look_outline(row: object) -> list | None:
    """DISPLAY-ONLY footprint outline for a crop overlay: the largest-area
    loop of the row's exact ``mask``, thinned shape-aware and rounded to
    2 dp. The overlay renderer draws ONE outline per mark; picking the
    dominant loop here is a rendering choice and feeds no relation/gap
    math (the relation substrate consumes the full multi-loop mask
    server-side). The normalization happens HERE — the single producer
    for the rendered overlay, the persisted manifest spec, and the offline
    re-render — so the pixels the model saw and the re-render input agree by
    construction (2 dp = 1 cm = 0.1 native px).

    Douglas-Peucker at half a cell, budget 256. The tolerance sits BELOW
    the 2 m cell quantization every prior footprint arrives at, so the
    stair-steps survive: they are the honest realisation of what the
    raster knows, and smoothing them would assert an edge the mask does
    not carry. The budget is a backstop — measured over Cal City's
    largest footprints (418k cells, 1,562 m across, 1,566-vertex ring)
    it never binds, and a rectangular building costs 4 vertices where
    the retired uniform rule spent 96 on redundant staircase corners."""
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
    center_boundary, annotate, marks}`` built by
    ``reasoner_loop._build_crop_requests`` off the round's ``_ns_rows``):
    resolve the ADAPTIVE window (fit the hull of the spec's own point set,
    floored at the subject's own scale and ceilinged at the stop's
    observation extent + the pixel budget — a digital crop of what was
    already captured; recenter on the marked-set centroid, edge-shift
    in-bounds with the subject-owns-its-window cap), then delegate the
    pixels to
    ``render_look_from_spec`` — the same deterministic tail the dashboard's
    re-render calls, so live and replay crops are byte-identical by
    construction.

    A near-raster-edge window shifts in-bounds to trim the black
    out-of-bounds border, but the shift is CAPPED so the requested fid always
    stays in ITS OWN window — a subject at/past the raster edge keeps a
    subject-centered window whose off-raster remainder renders as the black
    no-data border it truly is, with the validity facts
    (``window_valid_frac`` / ``subject_on_data``) riding the record so the
    caption states black-as-no-data.

    Returns one RECORD per crop (rank order), clamped to the 16-image server
    budget — ``{"jpeg": bytes}`` plus the resolved spec
    (``map_name/center_fid/center_label/center_boundary/marks/world_xy/
    intended_xy/pose_xy/context_m``) plus the derived audit fields
    (``sha256``, ``renderer_version``, ``edge_shift_m``, ``alt_m``,
    ``upscale``, ``gsd_m_per_px``, ``world_window``, ``marked``,
    ``identified``, ``outlined``, ``legend``, ``subject_in_window``,
    ``subject_centroid_in_window``, ``subject_on_data``,
    ``window_valid_frac``, ``marks_dropped``, ``marks_out_of_view``,
    ``marks_unbadged_in_frame``). The record minus ``jpeg`` IS the persisted
    manifest crop (``flight/look/...``) and IS the re-render spec:
    ``world_window`` is the SHIFTED window verbatim, so a post-hoc audit can
    still verify the crop CONTAINED the candidate / a known GT xy after an
    edge-shift. Pure of the reasoner — consumes only the specs + the block
    raster; a per-crop failure is logged and skipped, never raised."""
    from maps.perception.ortho.camera_model import (  # noqa: PLC0415
        ortho_gsd_m,
        ortho_world_bounds,
    )
    from maps.perception.ortho.ground_level import GROUND_LEVEL  # noqa: PLC0415

    if map_name not in GROUND_LEVEL:
        log.warning("look: map %r not in upstream GROUND_LEVEL; no crops",
                    map_name)
        return []

    # Raster world extent -> shift a near-edge crop window in-bounds so the
    # delivered image is mostly valid pixels. The flight-sim renders the
    # tile edge as a BLACK out-of-bounds border (a benchmark artifact, not a
    # real-world view); it wastes crop area and confuses the reader. A
    # candidate within half a context of the edge stays IN FRAME (just
    # off-center); one at/past the edge caps the shift instead (see the
    # keep-in-frame cap below) — the window then keeps its subject and the
    # caption states the black no-data truth. The record's ``world_window``
    # is the SHIFTED window verbatim so the GT-in-frame audit stays exact.
    _wb = None
    if imagery is None:
        try:
            _wb = ortho_world_bounds(map_name, rgbd_dir=rgbd_dir)
        except Exception as exc:  # noqa: BLE001 — no bounds -> just don't shift
            log.debug("look: no raster bounds for %s (%s); no edge-shift",
                      map_name, exc)
    # The run's imagery has no raster edge to shift away from: what it has
    # not imaged is black and said so, and the subject keeps its window.
    # The window bounds in metres, off this map's own grid spacing (the same
    # LRU-cached ortho the bounds above came from — no second load). An
    # unreadable ortho means no crops render at all, so the 0.1 m/px fallback
    # only ever sizes specs that then fail; it keeps the sizing pure rather
    # than raising from a renderer whose contract is to skip and log.
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
        # ADAPTIVE window: fit to the hull of everything the request names —
        # its center, and every mark's position AND FOOTPRINT — so however
        # many ids the agent passed are framed together, whole. A mark is an
        # OBJECT, so a window sized off its centroid alone frames a 78 m
        # compound in 66 m of ground and then declines to outline it for not
        # fitting; the shape is the thing being asked about, so the shape is
        # what the window has to hold. A lone subject (no marks) keeps its
        # subject-scaled floor.
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
        # below MIN); a maskless centroid-only row keeps the MIN floor.
        _diag = 0.0
        _bpts = [(float(p[0]), float(p[1]))
                 for p in (req.get("center_boundary") or [])
                 if isinstance(p, (list, tuple)) and len(p) == 2]
        if len(_bpts) >= 2:
            _bdx = max(p[0] for p in _bpts) - min(p[0] for p in _bpts)
            _bdy = max(p[1] for p in _bpts) - min(p[1] for p in _bpts)
            _diag = (_bdx * _bdx + _bdy * _bdy) ** 0.5
        # A POINT subject only — a bare [x, y] with no fid behind it. A
        # maskless FID (a car whose boundary did not ride the request) is
        # still a small thing and keeps the tight window; widening that
        # would bury it.
        _is_point = not str(req.get("center_fid") or "").strip()
        _floor = LOOK_POINT_CTX_M if (_is_point and _diag <= 0) else _min_m
        _ctx = max(_floor, 3.0 * _diag, _hull + LOOK_CAR_MARGIN_M)
        # CEILING 1 — the observation: a crop is a DIGITAL crop of the
        # stop's already-captured view square; you can crop less, not see
        # more. ``obs_span_m`` rides the request; absent = no observation
        # ceiling (offline/lab specs without a stop).
        _span = req.get("obs_span_m")
        if isinstance(_span, (int, float)) and _span > 0:
            _ctx = min(_ctx, max(_min_m, float(_span)))
        # CEILING 2 — the serving pixel budget (the constant's note).
        _ctx = min(_ctx, _budget_m)
        half_ctx = _ctx / 2.0
        # RECENTER on the requested point set (a fixed subject-centered window
        # would push a spread anchor out of frame + clip it at the mark loop
        # below); a lone target keeps its own xy.
        if len(pts) > 1:
            # The HULL CENTRE, never the arithmetic mean: the window is
            # SIZED off the hull, so only the hull's own centre is
            # guaranteed to contain it. The mean drifts toward whichever
            # side carries more points, and it clipped rows out of a window
            # already wide enough to hold them (measured: 24 rows spanning
            # 24 m in a 30 m window -- 16 drawn, 8 named out-of-view).
            ctr_x = (max(_xs) + min(_xs)) / 2.0
            ctr_y = (max(_ys) + min(_ys)) / 2.0
            # THE SUBJECT OWNS ITS WINDOW: cap the shift PER AXIS so it
            # keeps a half-car of margin inside each edge. This bites ONLY
            # when a ceiling clamped the window below the hull — then a far
            # anchor falls off (the legend still names it) rather than the
            # subject. Unclamped it is a provable no-op and needs no guard:
            # _ctx >= _hull + MARGIN makes _keep >= _hull/2, and the subject
            # is a point OF the hull, so it sits at most _hull/2 from the
            # hull centre on each axis.
            _keep = max(0.0, half_ctx - LOOK_CAR_MARGIN_M / 2.0)
            ctr_x = min(max(ctr_x, true_x - _keep), true_x + _keep)
            ctr_y = min(max(ctr_y, true_y - _keep), true_y + _keep)
        else:
            ctr_x, ctr_y = true_x, true_y
        pose_x, pose_y = resolve_window_center(
            (true_x, true_y), (ctr_x, ctr_y), _wb, half_ctx, LOOK_CAR_MARGIN_M)
        # The spec is built from the SAME rounded values it persists (2 dp =
        # 1 cm = 0.1 native px): rendering from the rounded spec is what
        # makes the manifest record a byte-exact re-render input rather than
        # an approximation of one.
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
    ``source: "run"``, so a re-render from the bake (``source: "bake"``)
    hashing differently is a read of which pixels a live look showed, not
    a renderer that moved.

    THE identity guarantee of the look audit: the live round
    (``render_candidate_crops``) and the dashboard's re-render (the render
    daemon's ``/render_look``) both call this function, and the manifest crop
    record IS the spec — so the pixels the model saw are reproducible from
    text + the local raster tree, and nothing stores jpeg bytes. The returned
    record's ``sha256`` fingerprints the exact delivered bytes; a re-render
    that hashes differently means the renderer or raster moved
    (``renderer_version`` / ``renderer_env`` say which), surfaced to the
    viewer, never silently swapped.

    Consumes ``{map_name, center_fid, center_label, center_boundary, marks,
    world_xy, intended_xy, pose_xy, context_m}``; extra keys (a full manifest
    record) are ignored and every derived field is recomputed. Returns the
    full crop record ``{"jpeg": bytes, ...}`` or None (logged) on a
    validation/render failure."""
    import hashlib  # noqa: PLC0415
    import cv2  # noqa: PLC0415 — hard dep of the overlay renderer
    import numpy as np  # noqa: PLC0415
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
    # The native render is already north-up (max world y at row 0), which
    # is the frame every op/bearing speaks — deliver it untouched; the
    # mark projector below shares the same row mapping.
    rgb = np.ascontiguousarray(oc["rgb"])
    validity = np.ascontiguousarray(oc["validity"])
    # Upscale toward the delivered-side target, derived from the native
    # side: a tight crop gains magnification (x3 at 15 m), a wide
    # relational frame ships native (x1); up*native stays within the
    # 1024 px budget side for every native width.
    _up = max(1, min(3, round(LOOK_TARGET_PX / max(1, rgb.shape[1]))))
    if _up > 1:
        rgb = cv2.resize(rgb, (rgb.shape[1] * _up, rgb.shape[0] * _up),
                         interpolation=cv2.INTER_LANCZOS4)
    H, W = rgb.shape[:2]
    # DECLARED-VIEW rotation (``frame_up`` spec key, a world [ux, uy] unit
    # from the same frame forms compare takes): rotate the delivered image
    # so that view's up is IMAGE-up. Pixels and ink rotate TOGETHER (the
    # projector below composes the same matrix), corners introduced by the
    # rotation render as the black no-data border the caption already
    # explains. ``validity`` keeps a pristine copy for the subject-on-data
    # point sample (rotation cannot change whether the subject sits on
    # data) and rotates for the delivered-frame valid fraction.
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
    # derived AUDIT geometry: the true candidate xy, the (possibly
    # edge-shifted) crop center, the shift magnitude, the window + gsd --
    # so a post-hoc audit can confirm the crop CONTAINED the candidate /
    # a known GT xy even when shifted.
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
        # edge_shift = how far the edge-clamp moved the crop off its
        # INTENDED center (the marked-set centroid), NOT the recenter
        # offset from the candidate (that is deliberate framing).
        "edge_shift_m": round(float(np.hypot(pose_x - ctr_x,
                                             pose_y - ctr_y)), 2),
        "alt_m": round(ground + half_ctx, 2),
        "context_m": _ctx,
        "upscale": _up,
        "gsd_m_per_px": round((x1 - x0) / max(W, 1), 4),
        # the delivered pixel grid: not constant (upscale is
        # clamp(round(450/native), 1, 3)), so the caption states it per crop
        "side_px": int(W),
        # what the CAMERA resolved, against which the line above is only
        # grid spacing over a resample factor
        "source_gsd_m": (round(float(imagery.gsd_m), 4) if imagery is not None
                         else _source_gsd_m(map_name)),
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
    # world-y, the native raster order; validated by the R3 raster
    # cross-check) — marks and image share the same row mapping.
    sx = (x1 - x0) or 1.0
    sy = (y1 - y0) or 1.0

    def _w2p(wx, wy, _x0=x0, _y1=y1, _sx=sx, _sy=sy, _W=W, _H=H):
        px = (float(wx) - _x0) / _sx * _W
        py = (_y1 - float(wy)) / _sy * _H
        if _rotM is not None:
            px, py = (_rotM[0, 0] * px + _rotM[0, 1] * py + _rotM[0, 2],
                      _rotM[1, 0] * px + _rotM[1, 1] * py + _rotM[1, 2])
        return px, py

    # CENTER IDENTIFICATION (every crop): the crop is centered on this
    # fid, so mark ITS car with the non-occluding mask-lasso mark
    # (footprint outline + digit badge; naming lives in the tool-text
    # legend). This is the fix for "clean crop, but WHICH car is <fid>?" --
    # the outline traces the recorded footprint, the color stays visible,
    # and the legend text stays clear of the subject footprint.
    center_px = _w2p(true_x, true_y)
    center_label = meta["center_label"]

    # Optional relational context: a distinct in-frame target + other
    # requested candidates, all achromatic v2 marks with digit badges
    # (1=center, 2=target, 3..=context) mirrored by the tool-text
    # legend. Marks exist iff the request asked for them; geometry
    # above already framed them.
    marks = req.get("marks") or None
    target_px: tuple[float, float] | None = None
    target_label = ""
    neighbor_px: list[tuple[float, float]] = []
    # ONLY the marks whose footprint actually reached the pixels: a mark
    # keeps its badge and its legend line when its shape does not fit the
    # frame (``_proj_poly``), and a record claiming an outline that was
    # never drawn is the audit lying about what the model saw.
    outlined: list[dict] = []

    def _proj_poly(boundary):
        """World boundary verts -> the OPEN crop-px arcs that lie in frame.

        A footprint bigger than its window still says something true about
        the part you can see, so it is CLIPPED, never dropped: the in-frame
        arcs draw and the crossings stay gaps. What must never draw is an
        edge running ALONG the frame edge — polygon clippers synthesise
        those, and one of them tells the reader the lake ends at the window
        boundary, which is a fact about the crop wearing the shape's
        clothes. ``clip_ring_to_box`` cuts edge by edge for that reason.

        Vertex fidelity belongs to the producer (``_look_outline`` thins
        shape-aware at half a cell), not to a fit test here. The failure
        this replaces was straight white chords across open water, and they
        came from thinning by INDEX — every n-th vertex of a raster ring,
        corners cut wherever the stride landed — not from the window being
        small.
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
        # its stroke rather than being shaved by the cut (vector-tile
        # practice: clip with a buffer, never on the nose).
        return clip_ring_to_box(pts, (-2.0, -2.0, W + 2.0, H + 2.0)) or None

    target_poly_px = None
    neighbor_polys_px: list = []
    # V2 LEGEND (lit: OmniParser 2408.00203 digit->description text
    # binding = +23.3pp): digits mirror the overlay's deterministic
    # assignment (1 = center, 2 = target, 3.. = neighbors in append
    # order); the tool text renders these lines. Density cap
    # LOOK_MAX_BADGES total (ordering measured flat to 25; center and
    # target keep reserved slots) — neighbors beyond the cap are
    # dropped WITH a legend note, never silently.
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
    _next_digit = [0]              # resolved after the loop, once the base
                                   # (2 with a target, 1 without) is known
    for mk in (marks or []):
        mxy = mk.get("xy") if isinstance(mk, dict) else None
        if not (isinstance(mxy, (list, tuple)) and len(mxy) == 2):
            continue
        px, py = _w2p(mxy[0], mxy[1])
        if not (0.0 <= px <= W and 0.0 <= py <= H):
            # The mark's CENTROID pixel is outside this crop — but a mark
            # is an OBJECT like the subject: only call it out-of-view when
            # centroid-OR-footprint containment fails too (the same
            # edge-clamp geometry that produced the 8267 false subject
            # warning; a footprint-in-frame mark just cannot carry a
            # centroid badge). The ack distinguishes the two cases.
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
        # Named places on the photograph, collected where the projection
        # already happened. ROADS only: a label on a BUILDING would cover
        # the evidence pixels the lasso exists to protect, which is the
        # whole reason this surface badges instead of chipping. One label
        # per NAME -- a street is many marks sharing one.
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
    # takes the next digit. The legend and the drawn badges must count the
    # same way or a caption names a badge that is not in the picture.
    # Neighbours start at 3 here: 1 = centre, 2 = target, which is the
    # overlay's own neighbor_digit_start default and what the legend has
    # always counted (2 + len(neighbor_px)). Not the map renderer's base,
    # which reassigns 2 when no target is present.
    _seq = 2
    _ctx = [_le for _le in legend if _le.get("role") == "context"]
    for _i, _slot in enumerate(_neighbor_digits):
        _le = _ctx[_i] if _i < len(_ctx) else None
        if _slot is None:
            # Identified by NAME: no ordinal drawn, and the legend says so
            # rather than advertising a badge the picture does not carry.
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
            center_label=center_label,
            center_poly_px=_proj_poly(req.get("center_boundary")),
            px_per_m=float(W) / max(x1 - x0, 1e-6),
            target_px=target_px,
            target_label=target_label,
            target_poly_px=target_poly_px,
            neighbor_polys_px=neighbor_polys_px,
            neighbor_px=neighbor_px,
            # a rotated crop's band states its TRUE up; "N up" only when
            # the pixels really are north-up.
            band_text=("N up" if _rotM is None
                       else f"up {round(up_bearing_deg(_fup)) % 360} deg"),
            # A named road is identified BY ITS NAME, so it takes no
            # ordinal: the badge would be a second identity for the same
            # object, and one that costs a legend lookup the name spares.
            neighbor_digits=_neighbor_digits,
            # Named places on the photograph. The cue is written in street
            # names and a badge-only crop makes the model leave the image
            # for the row table to learn WHICH road it is looking at. One
            # label per NAME, not per segment -- a street is many marks
            # sharing a name.
            place_labels=_place_labels,
        )
    except Exception as exc:  # noqa: BLE001
        # The center-fid identification IS the crop's visual identity
        # (which car it is of). A crop we cannot identify is worse than no
        # crop: measured, neutral no-bracket crops score BELOW not-looking
        # (the reader can't tell which car is the fid, so it confuses the
        # read). So a failed identification overlay SKIPS this crop rather
        # than delivering the harmful unidentifiable image.
        log.warning("look: identification overlay failed, skipping crop "
                    "at %s: %s", (true_x, true_y), exc)
        return None
    # HONESTY: the window that actually rendered vs the subject that
    # was requested. An edge-clamped/mis-resolved crop can exclude its
    # own subject (measured: three off-tile trio crops rendered black
    # under a caption promising them; a mis-resolved fid delivered a
    # window 122m away) — the caption must say so, not promise pixels.
    # The subject is an OBJECT: containment is centroid-OR-footprint
    # (subject_window_containment), so an elongated prior whose centroid
    # is off-window/off-tile while its footprint enters the frame is
    # truthfully IN the image (the caption states the partial case).
    _sin, _cin = subject_window_containment(
        meta["world_window"], (true_x, true_y),
        req.get("center_boundary"))
    meta["subject_in_window"] = _sin
    meta["subject_centroid_in_window"] = _cin
    # HOW BIG the subject is, and whether the frame holds all of it. The
    # three cases above all key on the subject being partly or wholly OUT;
    # none fires for the one that confuses a reader most, where the
    # centroid and part of the footprint are both in frame and the shape
    # runs kilometres past it in every direction. Cal City's
    # landuse=residential relation is 1,162 x 2,672 m against a 300 m
    # ceiling: the badge then binds its digit to two line fragments in a
    # corner, and nothing on the image or in the text says the thing
    # continues. A digit bound to an arbitrary fragment is worse than no
    # digit, so the caption states the extent and says the outline is cut.
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
    # RASTER-DATA truth: with the window subject-centered, part of it can
    # extend past the raster's data edge (validity 0 = no-data, rendered
    # black). The caption must state black-as-no-data, and — the sharper
    # case — that a subject whose POSITION is in the window can still
    # have NO pixels (it sits past the data edge inside the frame).
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
    # The delivered-bytes fingerprint: what the sha-verified audit doctrine
    # stores INSTEAD of the jpeg file. cv2.imencode output is deterministic
    # for one cv2 build; across builds pixels match and bytes may not —
    # the viewer's verdict states which.
    jp = bytes(jp)
    meta["sha256"] = hashlib.sha256(jp).hexdigest()
    return {"jpeg": jp, **meta}
