"""SAM 3.1 image adapter — server-only.

Per-frame POST to the SAM 3.1 server's ``/segment`` endpoint
(``serving/sam_server.py``). Every call runs the SAM 3.1
batched ``model(batch)`` forward: the vision encoder runs once per image
and every concept decoder runs inside that single forward. The client
receives bbox + bit-packed mask, projects to world coords, and extracts
visual attrs locally so the wire payload stays small. Stable object
identity across an episode comes from the world-space tracker, not from
a SAM session.

``SAM_VIDEO_SERVER_URL`` can override the video endpoint when it differs
from ``SAM_SERVER_URL``. Default is ``http://127.0.0.1:8772``.
"""

from __future__ import annotations

import base64
import itertools
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image

from maps.perception.base import DetectedObject
from maps.perception.sam.class_hint import resolve_class_hint
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.geometry.geolocate import (
    project_mask_to_world_ortho,
)
from maps.perception.observation import Observation
from maps.util.mask_codec import (
    decode_mask_crop,
    decode_mask_full,
    encode_mask_record,
)
from maps.util.mask_region import MaskRegion

DEFAULT_SAM_VIDEO_SERVER_URL = "http://127.0.0.1:8772"

# SAM 3.1's vision encoder ingests at this resolution after the server's
# RandomResize+pad transform. Sending a 4K frame over the wire wastes
# bandwidth (one PNG-encoded 3840×2160 frame is ~4-8 MB) when the
# encoder collapses it to 1008×1008 anyway. We aspect-preserve resize
# + letterbox-pad on the client so the wire payload is ~200-400 KB and
# the server sees identical pixels.
_SAM_INPUT_SIZE = 1008


class SamServerOutage(RuntimeError):
    """The SAM ``/segment`` server was UNREACHABLE for a whole detect call.

    OUTAGE SENTINEL (real-stack pivot, blocker V5-F2). A raised
    ``SamServerOutage`` means the segment server gave NO valid response at
    all — a transport fault, a 5xx, or a terminal ``None`` after the client
    exhausted its replica × retry budget (``_post_segment_avoid_stale_pool``
    returns ``None``). It is DISTINCT from a valid ``200`` carrying zero
    detections (a genuinely empty frame), which stays an empty list and is
    never an outage. The flight-path caller escalates this to the same
    infra-exclusion the compile/ingest outages use
    (``run_benchmark._exc_is_sam_infra`` → ``CITYNAV_EPISODE_INFRA_ABORT``),
    so the episode RE-QUEUES instead of silently scoring a false SR=0.
    """


def _endpoints() -> list[str]:
    """SAM video endpoint list.

    ``SAM_VIDEO_SERVER_URL`` wins when set, otherwise video calls reuse
    ``SAM_SERVER_URL``. A comma-separated value spreads stateless
    ``/segment`` calls across replicas.
    """
    raw = (
        os.environ.get("SAM_VIDEO_SERVER_URL")
        or os.environ.get("SAM_SERVER_URL")
        or DEFAULT_SAM_VIDEO_SERVER_URL
    )
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    return eps or [DEFAULT_SAM_VIDEO_SERVER_URL]


_rr = itertools.count()


def _pick_url() -> str:
    """Round-robin replica per call for the stateless ``/segment`` endpoint.

    A multi-replica ``SAM_VIDEO_SERVER_URL`` load-balances per call instead of
    funnelling every flight frame to one GPU (which idles the other SAM
    replicas and caps perception throughput at one replica). Round-robin, not
    random: a SAHI stop fires its tiles concurrently, and random assignment
    over R replicas leaves an expected max load well above ``tiles/R`` while
    other replicas sit idle. It also
    makes the poster's failover ``picker()`` advance OFF a dead replica rather
    than possibly re-drawing it.

    ``next`` on an ``itertools.count`` is atomic, so concurrent tiles get
    distinct indices without a lock.
    """
    eps = _endpoints()
    return eps[next(_rr) % len(eps)]


def tile_pool_width(n_tiles: int, configured: int | None) -> int:
    """How many SAHI tiles are in flight at once.

    ONE THREAD PER TILE when nothing is configured. The pool used to size
    itself off the replica list (``len(endpoints) + 1``), which leaves a TAIL
    WAVE whenever there are more tiles than that: the Cal City stop fires 9
    tiles at a width of 8, so the ninth waits for a whole wave of its own and
    the stop pays two round trips to do one tile's work. Measured at
    ``segment=1.4`` of a 3.5 s stop.

    Width costs threads, not GPU. Each one is a blocking HTTP call, and how
    much SAM work actually runs at once is the servers' own admission gate
    (``SAM_MAX_INFLIGHT``), not this number — so covering the tiles removes
    the tail without asking the plane for more than it already bounds.

    It cannot change what comes back: ``ex.map`` preserves bounds order, so
    tile order and per-tile wire order are the same at any width. A pinned
    ``1`` is still serial semantics, the triage knob.
    """
    width = configured or max(len(_endpoints()) + 1, n_tiles)
    return max(1, min(int(width), n_tiles or 1))


def _post_segment_avoid_stale_pool(*, picker, json_body) -> dict | None:
    """Stateless POST per call that bypasses the urllib3 connection pool.

    Three layers of pool-avoidance:

    1. ``with requests.Session()`` per call — empty pool at start, closed
       at end, so no idle socket ever lives across two calls in this
       process.
    2. ``HTTPAdapter(pool_connections=1, pool_maxsize=1)`` — a different
       module (e.g. a global ``Session`` someone else mounted) cannot
       inject a pooled socket; the per-call session owns its own
       single-slot pool.
    3. ``Connection: close`` — the server closes its half too, so even
       a tunnel/load-balancer hop that wants to cache a socket sees a
       proper FIN and doesn't try to reuse it next round.

    On top of that, ``Retry`` is configured to retry POST (``urllib3``'s
    default ``allowed_methods`` excludes POST, which is what causes the
    3-retry loop above to fail identically across attempts). 4xx surfaces
    immediately (client bug); transport errors and 5xx fail over to a
    fresh replica via ``picker()``.

    Returns the parsed JSON payload, or ``None`` on terminal failure.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # urllib3 sleeps backoff_factor * 2**(n-1) before retry n: 3, 6, 12 s,
    # 21 s across the three, so the ladder spans a radio dropout instead of
    # spending every retry inside one.
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"POST", "GET"}),
        backoff_factor=3.0,
        raise_on_status=False,
    )
    headers = {"Connection": "close"}
    last_exc: Exception | None = None
    for attempt in range(3):
        url = picker() + "/segment"
        try:
            with requests.Session() as s:
                adapter = HTTPAdapter(
                    max_retries=retry, pool_connections=1, pool_maxsize=1
                )
                s.mount("http://", adapter)
                s.mount("https://", adapter)
                resp = s.post(url, json=json_body, headers=headers,
                              timeout=(SEGMENT_CONNECT_TIMEOUT_S, SEGMENT_READ_TIMEOUT_S))
                if 400 <= resp.status_code < 500:
                    print(
                        f"[sam] /segment 4xx at {url}: "
                        f"{resp.status_code} {resp.text[:200]}",
                        flush=True,
                    )
                    return None
                resp.raise_for_status()
                return resp.json()
        except requests.exceptions.RequestException as e:
            last_exc = e
        if attempt < 2:
            time.sleep(0.5 * (attempt + 1))
    print(
        f"[sam] /segment call failed after 3 attempts "
        f"(stale-pool avoidance active): {last_exc}",
        flush=True,
    )
    return None


# Perception color enrichment. We request BARE-NOUN SAM concepts (one clean "car"
# layer, not "white car"/"gray car" sub-layers) and derive color HERE, over the mask
# pixels, as a structured attribute. This labels what the object LOOKS like (like the
# SAM class) -- it is NOT cue-language processing; the agent matches the instruction's
# color word to this percept by reasoning. We expose BOTH a coarse palette name AND
# the median RGB so the agent can settle fuzzy cases (silver vs gray) the name can't.
# Named color palettes (perceived-RGB anchors for nearest-name). "default" is
# ideal sRGB. A per-camera/environment calibration overrides the well-sampled
# anchors with the median PERCEIVED car color measured on that imagery — the
# ideal anchors are far too bright/saturated for dark, low-contrast aerial frames
# (so "white"/"red" never fire). Select per run via the adapter's
# ``color_palette`` field (YAML ``sam.color_palette``) or
# ``set_color_palette``; add a new entry to calibrate a new camera.
_COLOR_PALETTE_DEFAULT: dict[str, tuple[int, int, int]] = {
    "white": (235, 235, 235), "black": (32, 32, 32), "gray": (128, 128, 128),
    "silver": (190, 192, 198), "red": (170, 40, 40), "blue": (45, 70, 160),
    "green": (45, 130, 70), "yellow": (215, 205, 70), "orange": (220, 130, 45),
    "brown": (110, 75, 50), "tan": (200, 180, 145), "gold": (200, 165, 70),
}
# CityNav upstream birmingham/cambridge nadir-ortho rasters: the 5 well-sampled
# car colors recalibrated to the median perceived (SAM-mask interior) RGB,
# n>=15/color over the LLM-labeled GT set. MEASURED +10.8pp color-family /
# +10.4pp white<->black lightness (2-fold CV, held-out maps). Under-sampled
# colors keep the ideal anchor — EXCEPT anchors the calibration shows are
# INSIDE the calibrated cluster:
#
# "silver" is dropped from this palette. It is under-sampled, so it would keep
# the bright ideal (190,192,198), which sits ABOVE the recalibrated white
# (175,183,191) on this dark imagery — so most true white cars would snap to
# "silver", a word CityNav cues essentially never use, while the 8-unit
# white/silver margin crushes achromatic color_conf. A genuine silver car
# labeled "white" still matches a "silver" cue at 0.8 via the achromatic
# family — nothing is droppable by omitting it.
_COLOR_PALETTE_CITYNAV: dict[str, tuple[int, int, int]] = {
    k: v
    for k, v in {
        **_COLOR_PALETTE_DEFAULT,
        "white": (175, 183, 191), "gray": (121, 132, 145), "black": (77, 86, 98),
        "red": (135, 78, 88), "blue": (75, 101, 136),
    }.items()
    if k != "silver"
}
_COLOR_PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "default": _COLOR_PALETTE_DEFAULT,
    "citynav": _COLOR_PALETTE_CITYNAV,
}
_ACTIVE_PALETTE_NAME: str = "default"


def set_color_palette(name: "str | None") -> None:
    """Select the active color palette by name (e.g. ``"citynav"``). ``None`` /
    empty keeps the current palette so a config that doesn't
    specify one doesn't clobber an earlier selection; an unknown
    name falls back LOUDLY to ``"default"`` (a config typo must not silently
    disable the calibration). Module-global so BOTH the detection path and the
    prior-percept enrichment path (which calls ``_set_color_attr`` directly)
    share one calibration; the resolved name is printed for provenance."""
    global _ACTIVE_PALETTE_NAME
    if not name:
        return
    if name not in _COLOR_PALETTES:
        print(
            f"[sam] WARNING: unknown color_palette {name!r}; "
            f"known={sorted(_COLOR_PALETTES)} — using 'default' (ideal sRGB; the "
            f"calibrated color lever is OFF)",
            flush=True,
        )
        _ACTIVE_PALETTE_NAME = "default"
        return
    _ACTIVE_PALETTE_NAME = name
    print(f"[sam] active color_palette = {name}", flush=True)


def _active_palette() -> dict[str, tuple[int, int, int]]:
    return _COLOR_PALETTES.get(_ACTIVE_PALETTE_NAME, _COLOR_PALETTE_DEFAULT)


# Sample the INTERIOR (center) pixels for color, not the whole mask. A small
# nadir car mask is ~¼–½ boundary pixels that straddle the road/asphalt; those
# mixed pixels drain chroma and darken the median (colorfulness ∝ channel
# std-dev). Keeping the inner pixels by distance-from-edge moves the median onto
# the body (roof). MEASURED on GT cars (perceived-vs-cue color family, n≈39–50
# over 3 match radii): +14–17pp accuracy vs the full mask, driven by the
# achromatic majority — and it is the *only* selection step that helped (a
# two-sided value gate was inert, white-balance was negative). Pixel selection,
# not the averaging statistic, is the measured lever (Hsieh 2015; Stavrothanasopoulos 2022).
_COLOR_INTERIOR_PCTL = 50
_COLOR_INTERIOR_MIN_PX = 12


def _interior_mask(region) -> "np.ndarray":
    """Inner pixels of ``mask`` (distance-from-edge >= the ``_COLOR_INTERIOR_PCTL``
    percentile), to drop the road-contaminated boundary ring. Falls back to the
    full mask when erosion would leave too few pixels (a tiny detection).

    Returns the selection LOCAL to the region's crop, so the caller indexes
    ``region.window(rgb)[inner]`` — row-major over the same pixels a
    whole-frame ``rgb[inner]`` walked, in the same order, with no frame-sized
    selection ever built."""
    from scipy import ndimage

    m_box = region.crop
    if int(m_box.sum()) < _COLOR_INTERIOR_MIN_PX:
        return m_box
    sub = np.pad(m_box, 1)  # 1px False border so the boundary registers
    d = ndimage.distance_transform_edt(sub)
    if d.max() <= 1.0:
        return m_box
    inner_sub = (d >= np.percentile(d[sub], _COLOR_INTERIOR_PCTL)) & sub
    inner_box = inner_sub[1:-1, 1:-1]
    return inner_box if int(inner_box.sum()) >= _COLOR_INTERIOR_MIN_PX else m_box


# Per-color reliability PRIOR for the per-detection color CONFIDENCE. SEED values
# only: chromatic car colors read UNRELIABLY off dark, low-contrast nadir masks
# (measured ~57-80% mislabel on chromatic sole-cues red/blue), achromatic reliably
# (~16-19% mislabel for white/gray/black/silver). The real per-color table is FIT on
# train_seen LATER (NOT on eval episodes -- that would leak); these seeds keep the
# score sane until then. Achromatic family enumerated locally to avoid importing the
# maps.search palette (no cross-package layering).
_COLOR_RELIABILITY_PRIOR_DEFAULT = 0.35  # chromatic seed
_COLOR_ACHROMATIC = frozenset({"white", "black", "gray", "silver"})


def _color_reliability_prior(name: str) -> float:
    return 0.82 if name in _COLOR_ACHROMATIC else _COLOR_RELIABILITY_PRIOR_DEFAULT


def _dominant_color(rgb: "np.ndarray | None", region):
    """Median color over the masked pixels -> (name, [r, g, b], conf), or None.

    Sampled over the mask's INTERIOR (``_interior_mask``) so the road-contaminated
    boundary ring does not darken/desaturate the median. Robust by construction:
    returns None on missing/shape-mismatched/empty inputs so it can NEVER crash the
    detection path. Median (not mean) is robust to windshield/shadow outliers.
    Nearest-palette naming is a perception labeling step (not parsing the operator's
    language)."""
    if region is not None and not isinstance(region, MaskRegion):
        # Same reasoning as color_from_mask_m6: the except below is broad on
        # purpose so colour cannot kill a flight, and that breadth already hid
        # one un-migrated caller (osm_prior's, reached through a `_sca` alias
        # that a grep for the real name missed) as colour silently absent.
        raise TypeError(
            f"_dominant_color takes a MaskRegion, got {type(region).__name__}"
        )
    try:
        if rgb is None or region is None:
            return None
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or tuple(rgb.shape[:2]) != tuple(region.frame_shape):
            return None
        inner_box = _interior_mask(region)
        if not inner_box.any():
            return None
        px = region.window(rgb)[inner_box][:, :3].astype(np.float32)
        if px.shape[0] == 0:
            return None
        med = np.median(px, axis=0)
        dists = sorted(
            (float(np.sum((med - np.asarray(anchor, dtype=np.float32)) ** 2)), cname)
            for cname, anchor in _active_palette().items()
        )
        name = dists[0][1]
        # RGB-MARGIN confidence: how cleanly the median snaps to ONE palette anchor vs
        # the runner-up. dists are SQUARED euclidean; sqrt -> actual RGB distance.
        # margin in [0,1]: ~0 when the color sits between two anchors (an ambiguous
        # label, i.e. exactly the chromatic mislabel case), ~1 when it snaps cleanly.
        # x the per-color reliability prior. Parameter-free (no temperature).
        d1 = float(np.sqrt(max(0.0, dists[0][0])))
        d2 = float(np.sqrt(max(0.0, dists[1][0]))) if len(dists) > 1 else d1
        margin = (d2 - d1) / (d2 + d1) if (d1 + d2) > 0.0 else 0.0
        conf = round(max(0.0, min(1.0, margin * _color_reliability_prior(name))), 3)
        return name, [int(med[0]), int(med[1]), int(med[2])], conf
    except Exception:
        return None


def _set_color_attr(attrs: dict, rgb, region) -> None:
    """Populate ``attrs['color']`` (coarse name) + ``attrs['color_rgb']`` (median RGB)
    from the region's pixels, if derivable. No-op on failure."""
    col = _dominant_color(rgb, region)
    if col is not None:
        attrs["color"] = col[0]
        attrs["color_rgb"] = f"{col[1][0]},{col[1][1]},{col[1][2]}"
        # Stringified float (det.attributes is dict[str,str]; mirrors color_rgb). The
        # graph folds it into semantic_attributes as {value:count} like every attr;
        # the model reads it back off the surface row.
        attrs["color_conf"] = f"{col[2]:.3f}"








# Per-instance mask split. SAM 3.x emits ONE mask per DETR query with NO
# instance separation / NO NMS / NO connected-component split (paper §3
# arXiv 2511.16719; HF post_process applies none). For tiny, dense objects a
# single 'instance' mask can therefore span multiple DISJOINT objects (two
# distant cars under one mask) or carry stray speckle. We split each returned
# mask into connected components so each real instance gets its own world_xy +
# perceived color. The noise floor is RELATIVE to the class's typical
# single-instance size (self-calibrating across altitude/GSD and class), so it
# needs no per-class physical prior; at this benchmark's GSD ~0.25 m/px,
# ``_INSTANCE_SPLIT_AREA_FRAC = 0.10`` of the median car mask ≈ a ~1 m² floor.
_INSTANCE_SPLIT_AREA_FRAC = 0.10
_INSTANCE_SPLIT_MIN_REF_SAMPLES = 5
# A non-largest component is REDUNDANT (a worse partial of a neighbor that has
# its own, better-cropped detection) when more than this fraction of its pixels
# is already covered by ANOTHER same-class detection. Dropped so the split does
# not mint a duplicate instance of an already-detected car.
_INSTANCE_SPLIT_REDUNDANT_FRAC = 0.5


def _connected_components(region: MaskRegion) -> "list[MaskRegion]":
    """8-connected components of a region, as regions over the same frame.

    Returns ``[region]`` UNCHANGED when single-component (the common case), and
    that identity is load-bearing: ``_split_and_pack`` keys ``comp is region``
    off it to know the mask was not split.

    Labelling runs on the crop, which is the whole point of the region: every
    set pixel is inside the box, so 8-connectivity between any two of them is
    unchanged, and ``ndimage.label`` numbers components in raster order, which
    the crop preserves. Component k is the same component, in the same list
    position — and that position is load-bearing too, since
    ``max(comps, key=area)`` returns the FIRST maximum.
    """
    from scipy import ndimage

    labels, n = ndimage.label(region.crop, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return [region]
    # Each component re-boxes to its own extent, so a stray speckle far from
    # the body does not drag the whole parent box along behind it.
    return [region.with_crop(labels == i).tightened() for i in range(1, n + 1)]


def _wire_mask_region(
    record: dict,
    *,
    wire_hw: tuple[int, int],
    scale: float,
    pad_top: int,
    pad_left: int,
    frame_hw: tuple[int, int],
) -> "MaskRegion | None":
    """Bring a wire mask into FRAME coordinates.

    The wire carries a bbox crop in the letterboxed frame the server saw. Two
    geometrically different cases, not a fast path and a slow one:

    * **Identity letterbox** — the SAHI path re-packs its NMS'd masks into a
      payload whose wire frame IS the full frame (``scale=1``, no pads), so the
      wire box is already a frame box and the crop needs no resampling. This is
      the hot path, and it used to build a frame-sized canvas, multiply it by
      255, NEAREST-resize it onto itself and threshold it — four whole-frame
      operations that were all no-ops.
    * **Real letterbox** — the direct ``/segment`` path resizes and pads, and a
      NEAREST resize does not decompose per box (an output row's source row
      depends on the whole extent), so the frame is materialised once and the
      region cropped back out of it. Correctness first; this path is not the
      one under per-detection pressure.

    ``None`` when the mask lands empty after clipping.
    """
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    new_h = max(1, int(round(frame_h * float(scale))))
    new_w = max(1, int(round(frame_w * float(scale))))
    identity = (
        float(scale) == 1.0
        and int(pad_top) == 0
        and int(pad_left) == 0
        and (new_h, new_w) == (frame_h, frame_w)
        and (int(wire_hw[0]), int(wire_hw[1])) == (frame_h, frame_w)
    )
    if identity:
        r0, c0, r1, c1 = (int(v) for v in record["pixel_bbox_rcrc"])
        # Clip to the frame exactly as decode_mask_full does, so a record from
        # a differently-sized frame places what it can rather than raising.
        r1 = min(r1, frame_h - 1)
        c1 = min(c1, frame_w - 1)
        if r1 < r0 or c1 < c0:
            return None
        crop = decode_mask_crop(record)[: r1 - r0 + 1, : c1 - c0 + 1]
        if not crop.any():
            return None
        return MaskRegion(crop, (r0, c0, r1, c1), (frame_h, frame_w))

    wire_full = decode_mask_full(record, (int(wire_hw[0]), int(wire_hw[1])))
    inner = wire_full[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    if inner.shape[0] != new_h or inner.shape[1] != new_w:
        # Defensive: keep arrays consistent under odd shapes.
        inner = inner[:new_h, :new_w]
    full = (
        np.asarray(
            Image.fromarray(inner.astype(np.uint8) * 255).resize(
                (frame_w, frame_h), resample=Image.Resampling.NEAREST
            )
        )
        > 0
    )
    return MaskRegion.from_full(full)


def split_into_instances(
    region: MaskRegion,
    *,
    ref_area_px: float | None = None,
    min_area_px: int = 16,
    frac: float = _INSTANCE_SPLIT_AREA_FRAC,
    coverage_count: "np.ndarray | None" = None,
    redundant_frac: float = _INSTANCE_SPLIT_REDUNDANT_FRAC,
) -> "list[MaskRegion]":
    """Split a SAM concept mask into per-instance connected-component masks.

    The common single-component case returns ``[mask]`` unchanged (a clean
    detection is NEVER split or filtered). For a multi-component mask the
    LARGEST component is always kept as the primary instance; each other
    component is then either kept or dropped:

    - **drop speckle** — area below the noise floor ``max(min_area_px, frac *
      ref_area_px)`` px, where ``ref_area_px`` is the median area of clean
      single-component same-class masks in the frame (None when too few clean
      samples → absolute ``min_area_px`` floor only);
    - **drop redundant** — when ``coverage_count`` (per-pixel count of OTHER
      same-class detections' KEPT PRIMARIES, i.e. the sum of every detection's
      largest component) is given and more than ``redundant_frac`` of the
      component's pixels have ``coverage_count >= 1``: another detection's
      primary already owns this region, so the component is a worse partial of a
      neighbor that has its own, better-cropped id. Keying off PRIMARIES (not
      raw mask overlap) is what makes a small car that is NOT the largest in any
      mask — covered only by *merged* masks, never claimed as a primary — survive
      (coverage 0) instead of being silently erased by a symmetric double-merge;
    - **keep** otherwise — a unique component nothing else detected (a missed
      car, even if the mask is only a rough sliver: a candidate near the car
      beats none).

    Never returns empty (the largest component is always kept). v1 does NOT
    re-merge near components."""
    comps = _connected_components(region)
    if len(comps) <= 1:
        return comps
    floor = int(min_area_px)
    if ref_area_px is not None and ref_area_px > 0:
        floor = max(floor, int(round(frac * float(ref_area_px))))
    order = sorted(comps, key=lambda c: c.area, reverse=True)
    cov = None if coverage_count is None else np.asarray(coverage_count)
    kept = [order[0]]  # largest = the detection's primary instance, always kept
    for c in order[1:]:
        if c.area < floor:
            continue  # speckle
        # This component is never covered by its OWN detection's primary
        # (disjoint components), so cov>=1 means ANOTHER detection's primary
        # owns it (redundant). A genuinely-unique car has cov 0 here and is kept.
        # Read through the component's own window: the same pixels in the same
        # order a frame-sized ``cov[c]`` walked.
        if cov is not None and float((c.window(cov)[c.crop] >= 1).mean()) > redundant_frac:
            continue  # redundant sliver of an already-detected neighbor
        kept.append(c)
    return kept


# zlib effort for the PNG that carries a frame to SAM. PIL defaults to 6,
# which costs 259 ms on a real 1008x1008 native tile against 79 ms at level 1,
# for 0.4% more bytes (1811.8 -> 1819.1 KB, measured on the birmingham_block_1
# upstream raster). A SAHI stop encodes 9 tiles, so the default was 2.40 s of
# client CPU per stop against 0.71 s, inside a ~7.4 s stop whose SAM service
# time is only ~1.3 s.
#
# Level, not format. Every PNG level is LOSSLESS, so SAM sees identical pixels
# and detections cannot move. JPEG would be ~20x faster still and was measured
# on the same tiles: SAM is bit-deterministic (a PNG re-run matches 734/734 at
# 0.00 px), and against that control JPEG q95 changed 12% of detections and
# 7.4% of cars. Lossy encode on this wire needs a paired SR read, not a
# latency argument.
_PNG_WIRE_LEVEL = 1


def _png_wire_bytes(canvas: np.ndarray) -> bytes:
    """Encode a letterboxed frame for the ``/segment`` wire.

    cv2 rather than PIL: measured on a real 1008 tile at the same level, cv2
    encodes in 65.8 ms against PIL's 78.0 and produces slightly SMALLER bytes
    (2337 vs 2361 KB) — better on both axes, at every level.

    ``[:, :, ::-1]`` is load-bearing, not tidying. cv2 writes its input as BGR,
    and the server decodes with PIL as RGB, so encoding the RGB canvas
    unreversed would hand SAM channel-swapped pixels — a change that decodes
    cleanly, looks like a valid image, and moves every detection.
    ``test_wire_png_is_lossless`` decodes through PIL for exactly that reason.
    """
    import cv2

    ok, buf = cv2.imencode(
        ".png", canvas[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, _PNG_WIRE_LEVEL]
    )
    if not ok:
        raise RuntimeError("cv2 refused to PNG-encode the SAM wire frame")
    return buf.tobytes()


#: How long the direct /segment call waits to connect: it outlasts a radio
#: dropout (about twenty seconds).
SEGMENT_CONNECT_TIMEOUT_S = 45
#: How long the direct /segment call waits for the server's answer. The
#: server holds a request behind its in-flight semaphore for
#: ``serving/sam_server.py``'s ``_INFLIGHT_WAIT_S`` (600 s) before
#: it answers 503; the client must outwait that, or it abandons a request
#: the server is still going to serve and posts the tile again on top.
#: ``maps/perception/tests/test_detect_client.py`` pins the order.
SEGMENT_READ_TIMEOUT_S = 660


def _letterbox_to_sam(rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize + center-pad to ``_SAM_INPUT_SIZE``.

    Returns ``(canvas, scale, pad_top, pad_left)`` where ``canvas`` is
    the 1008×1008 ``uint8`` letterboxed image, ``scale`` is the linear
    factor mapping original pixel coords to letterboxed-frame pixel
    coords (i.e. ``new = scale*orig + pad``), and ``pad_top`` /
    ``pad_left`` are the black-bar offsets in the letterboxed frame.
    Use the inverse to map detection coords back to the original frame.
    """
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return (
            np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), dtype=np.uint8),
            1.0,
            0,
            0,
        )
    if h == w == _SAM_INPUT_SIZE:
        # Native tile already exactly the model input -> feed 1:1, no
        # resample / pad / canvas copy (identity letterbox). The SAHI
        # native-tiling path relies on this to hand SAM real native-resolution
        # pixels instead of an interpolated downscale.
        return np.ascontiguousarray(rgb, dtype=np.uint8), 1.0, 0, 0
    scale = float(_SAM_INPUT_SIZE) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(rgb.astype(np.uint8)).resize(
            # LANCZOS both directions: Pillow scales the filter support to the
            # factor, so this is the anti-aliased shrink (the >1008 sensor
            # frames made the downscale branch real, not just the identity's
            # neighbour).
            (new_w, new_h), resample=Image.Resampling.LANCZOS
        )
    )
    canvas = np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), dtype=np.uint8)
    pad_top = (_SAM_INPUT_SIZE - new_h) // 2
    pad_left = (_SAM_INPUT_SIZE - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, pad_top, pad_left


@dataclass
class SamDetectClient:
    """Adapter onto the SAM 3.1 image server (stateless ``/segment``).

    One call surface: ``detect()`` — single frame, POSTs ``/segment``.
    Cross-frame object identity comes from the world-space tracker, not
    a SAM session.

    Every call ships the full unified concept list per the
    SAM-batched-concept rule: the encoder runs once and every concept
    decoder runs in the same forward.
    """

    concept_provider: ReasonerUnionConceptProvider
    model_id: str = "facebook/sam3.1"
    score_threshold: float = 0.3
    min_area_px: int = 16
    max_per_step: int = 64
    # Named color calibration for this camera/environment (see _COLOR_PALETTES).
    # None keeps the env/default palette; the CityNav configs set "citynav".
    # _build_perception activates it module-globally after construction.
    color_palette: "str | None" = None
    # Native-resolution SAHI tiling for the dense-small-object (ortho
    # ortho) path. When True, detect() tiles obs.rgb into
    # fixed _SAM_INPUT_SIZE (1008) tiles fed to SAM 1:1 (no resample), remap to
    # full-frame, and cross-tile IoU-NMS dedup -- recovering dense-lot car recall
    # that full-frame SAM loses to sub-pixel cars (~67%->82% measured). Pair with
    # a native-resolution render (run_benchmark rgb_upscale_px=2016) so the tiles
    # carry real 0.1 m/px detail rather than an upscaled 1008 frame.
    sahi: bool = False
    sahi_overlap: float = 0.2
    sahi_nms_iou: float = 0.5
    # Client-side concurrency. Tile workers hide PNG encode + RTT behind the
    # server's serial GPU; decode workers parallelize the per-detection mask
    # decode / projection / color CPU. Both preserve serial output order
    # exactly; 1 = serial semantics (fleet-side triage knob, settable via
    # YAML sam:).
    #
    # None = track the replica count: fan-out saturates near replicas+1
    # (measured, sam_fanout_bench: width 4 over 3 replicas is 1.05 s against
    # 1.03 s at width 9), so the width follows SAM_SERVER_URL rather than
    # sitting at a number that goes stale the next time replicas move. A
    # 4-wide client in front of 7 replicas leaves three idle.
    sahi_tile_workers: int | None = None
    decode_workers: int = 4

    def detect(
        self,
        obs: Observation,
        pose_xy: tuple[float, float],
        instruction_hint: str | None = None,
    ) -> list[DetectedObject]:
        """Stateless image-mode detection via ``POST /segment``.

        No session_id, no masklet tracking. Use for disjoint frames —
        the quick-parallel sweep over precomputed anchor cursors is the
        canonical caller, where the drone teleports between cursors and
        cross-frame tracking would just produce noise.
        """
        del pose_xy  # obs.ortho_view_corners_xy drives the projection
        if obs.rgb is None:
            return []
        if obs.ortho_view_corners_xy is None:
            raise ValueError(
                "SamDetectClient.detect requires obs.ortho_view_corners_xy "
                "(the view square a mask projects through)"
            )
        concepts_list = self.concept_provider.concepts_for(obs, instruction_hint)
        if not concepts_list:
            return []
        return self._segment(obs, list(concepts_list))

    def _segment(
        self, obs: Observation, concepts: list[str]
    ) -> list[DetectedObject]:
        """Dispatch one image-mode detection: native SAHI-tiled on the ortho
        path whenever the frame is larger than the encoder input (or ``sahi``
        forces it), else a single full-frame ``/segment``.

        Tiling by size is the DEFAULT: a frame over ``_SAM_INPUT_SIZE`` in
        either dimension would otherwise be letterbox-shrunk and hand the
        encoder a fraction of its pixels (a 2048×1536 sensor frame keeps 24%).
        ``_tile_bounds`` returns a single bound for small frames, so nothing
        changes below the threshold; ``sahi`` remains the explicit override
        for forcing tiles on frames that already fit."""
        h, w = obs.rgb.shape[:2]
        if self.sahi or h > _SAM_INPUT_SIZE or w > _SAM_INPUT_SIZE:
            return self._sahi_segment(obs, concepts)
        return self._segment_image_via_server(obs, concepts)

    @staticmethod
    def _tile_bounds(
        H: int, W: int, overlap: float
    ) -> "list[tuple[int, int, int, int]]":
        """Fixed native-size tiling: every tile is EXACTLY ``_SAM_INPUT_SIZE`` so
        it is fed to SAM 1:1 (no resample, no data discarded). The tile COUNT is
        DERIVED from the frame size + overlap, not a prescribed grid -- a larger
        (native-res) render simply yields more tiles, each still native. A 2016
        render gives a 3x3 grid (starts {0, 504, 1008})."""
        t = _SAM_INPUT_SIZE

        def starts(size: int) -> "list[int]":
            if size <= t:
                return [0]
            step = max(1, int(round(t * (1.0 - overlap))))
            n = max(2, -(-(size - t) // step) + 1)  # ceil((size-t)/step) + 1
            return [int(round(i * (size - t) / (n - 1))) for i in range(n)]

        rs = starts(H)
        cs = starts(W)
        return [(r, min(r + t, H), c, min(c + t, W)) for r in rs for c in cs]

    @staticmethod
    def _nms_masks(
        items: "list[tuple[np.ndarray | None, tuple[int, int, int, int], int, str, float]]",
        iou_thr: float,
    ) -> "list[tuple[np.ndarray, tuple[int, int, int, int], str, float]]":
        """Greedy IoU-NMS over bbox-cropped masks (highest score wins),
        bbox-prefiltered so it stays ~linear when masks are spatially disjoint.
        Items carry ``(crop, bbox_rcrc, area, label, score)`` with the bbox in
        full-frame coordinates; pixel work runs on the bbox-intersection window
        only, so nothing materializes at frame size. A ``None`` crop is an
        empty mask carried for the tile-det count — dropped here, exactly as
        the full-frame path dropped zero-pixel masks. This is the cross-tile
        seam dedup; the world-space tracker dedups again downstream by
        ``track_id``, so seam duplicates collapse twice."""
        order = sorted(range(len(items)), key=lambda i: -items[i][4])
        kept: list[tuple] = []  # (crop, (r0,c0,r1,c1), area, label, score)
        for i in order:
            crop, bb, a, lab, sc = items[i]
            if crop is None:
                continue
            dup = False
            for kcrop, kbb, ka, _, _ in kept:
                if bb[2] < kbb[0] or bb[0] > kbb[2] or bb[3] < kbb[1] or bb[1] > kbb[3]:
                    continue  # bboxes disjoint -> no overlap
                wr0, wr1 = max(bb[0], kbb[0]), min(bb[2], kbb[2])
                wc0, wc1 = max(bb[1], kbb[1]), min(bb[3], kbb[3])
                inter = int(
                    (
                        crop[
                            wr0 - bb[0] : wr1 - bb[0] + 1,
                            wc0 - bb[1] : wc1 - bb[1] + 1,
                        ]
                        & kcrop[
                            wr0 - kbb[0] : wr1 - kbb[0] + 1,
                            wc0 - kbb[1] : wc1 - kbb[1] + 1,
                        ]
                    ).sum()
                )
                if inter and inter / (a + ka - inter) > iou_thr:
                    dup = True
                    break
            if not dup:
                kept.append((crop, bb, a, lab, sc))
        return [(crop, bb, lab, sc) for (crop, bb, _a, lab, sc) in kept]

    def _sahi_segment(
        self, obs: Observation, concepts: list[str]
    ) -> list[DetectedObject]:
        """Native-resolution slicing-aided detection: tile ``obs.rgb`` into fixed
        ``_SAM_INPUT_SIZE`` tiles, ``/segment`` each at 1:1 (``_letterbox_to_sam``
        is identity at 1008 -- no resample), carry each mask as its bbox crop in
        full-frame coordinates (never materialized at frame size), cross-tile
        IoU-NMS dedup, then re-pack into a full-frame wire payload and run the
        SAME decode / split / projection / color path as the single-frame call.
        Recovers dense-lot car recall full-frame SAM loses to sub-pixel cars
        (~67%->82% measured on Birmingham dense lots)."""
        rgb = obs.rgb
        assert rgb is not None
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        # (crop, bbox_rcrc, area, label, score); crop None = empty mask
        tiledets: list[tuple] = []
        n_failed = 0
        bounds = self._tile_bounds(H, W, self.sahi_overlap)

        sam_body = {
            "concepts": concepts,
            "score_threshold": float(self.score_threshold),
            "min_area_px": int(self.min_area_px),
            "max_per_image": int(self.max_per_step),
            "resolution": _SAM_INPUT_SIZE,
        }
        def _one_tile(i: int) -> list[tuple] | None:
            """Fetch + decode one tile. Returns that tile's tiledets in wire
            order, or None on poster failure. Self-contained: reads only
            shared-immutable state (rgb, concepts, self thresholds), so tiles
            run concurrently; collection below stays in bounds order."""
            bound = bounds[i]
            r0, r1, c0, c1 = bound
            sub = rgb[r0:r1, c0:c1]
            ch, cw = int(sub.shape[0]), int(sub.shape[1])
            canvas, scale, pad_top, pad_left = _letterbox_to_sam(sub)
            json_body = {
                **sam_body,
                "image_b64": base64.b64encode(_png_wire_bytes(canvas)).decode("ascii"),
            }
            payload = _post_segment_avoid_stale_pool(
                picker=_pick_url, json_body=json_body
            )
            if payload is None:
                # A genuine tile failure (the poster exhausted its replica x retry
                # budget), NOT "no detections". Skipping it silently would blind
                # ~1/len(bounds) of the frame -- dense-lot cars vanish with no
                # signal, and a partial frame looks complete to the reasoner.
                # Count it and surface it loudly below (no-silent-failures rule).
                return None
            wire_H, wire_W = payload.get(
                "image_shape_hw", [_SAM_INPUT_SIZE, _SAM_INPUT_SIZE]
            )
            new_h = max(1, int(round(ch * scale)))
            new_w = max(1, int(round(cw * scale)))
            # Native tile: identity letterbox -> the unpacked tile-local mask is
            # already at full tile resolution, so skip the inverse NEAREST resize.
            native = (
                scale == 1.0
                and pad_top == 0
                and pad_left == 0
                and (new_h, new_w) == (ch, cw)
            )
            tile_out: list[tuple] = []
            for det in payload.get("detections", []):
                wire = decode_mask_full(det, (wire_H, wire_W))
                inner = wire[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
                if native:
                    tile_full = inner
                else:
                    tile_full = (
                        np.asarray(
                            Image.fromarray(inner.astype(np.uint8) * 255).resize(
                                (cw, ch), resample=Image.Resampling.NEAREST
                            )
                        )
                        > 0
                    )
                area = int(tile_full.sum())
                if area < self.min_area_px:
                    continue
                lab, sc = str(det["label"]), float(det["score"])
                rows = tile_full.any(axis=1)
                if not rows.any():
                    # zero-pixel mask (min_area_px == 0): counted as a
                    # tile-det, dropped by NMS like any empty mask.
                    tile_out.append((None, (0, 0, 0, 0), 0, lab, sc))
                    continue
                cols = tile_full.any(axis=0)
                lr0 = int(rows.argmax())
                lr1 = int(tile_full.shape[0] - 1 - rows[::-1].argmax())
                lc0 = int(cols.argmax())
                lc1 = int(tile_full.shape[1] - 1 - cols[::-1].argmax())
                # .copy() releases the decoded wire-frame array the slice views
                crop = tile_full[lr0 : lr1 + 1, lc0 : lc1 + 1].copy()
                tile_out.append(
                    (crop, (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1), area, lab, sc)
                )
            return tile_out

        # ex.map preserves bounds order, so tiledets insertion order (bounds
        # order, then per-tile wire order) is byte-identical to the serial
        # loop -- NMS stable-sort tie-breaking and repack object_ids depend
        # on it. A 1-worker pool is serial semantics (the triage knob).
        n_workers = tile_pool_width(len(bounds), self.sahi_tile_workers)
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="sahi-tile"
        ) as ex:
            per_tile = list(ex.map(_one_tile, range(len(bounds))))
        for tile_out in per_tile:
            if tile_out is None:
                n_failed += 1
            else:
                tiledets.extend(tile_out)
        if bounds and n_failed == len(bounds):
            # TOTAL outage: EVERY tile failed to get a usable response (each
            # returned ``None`` after retries). That is a whole-frame SAM
            # outage, not a partial-blind frame with real detections elsewhere,
            # and not a clean empty frame (a 200-with-zero-detections tile does
            # NOT count toward ``n_failed``). Escalate so the episode re-queues
            # instead of scoring a false SR=0 (blocker V5-F2). A PARTIAL failure
            # (0 < n_failed < len(bounds)) still surfaces loudly below and keeps
            # the real detections from the tiles that DID answer.
            raise SamServerOutage(
                f"SAM SAHI outage: all {len(bounds)} tiles failed "
                "(/segment returned no usable response after retries) — "
                "whole-frame outage, not a partial or empty frame"
            )
        kept = self._nms_masks(tiledets, self.sahi_nms_iou)
        # Re-pack the merged masks into a wire payload whose "wire frame" IS
        # the full frame (identity letterbox), so the existing decode (split +
        # projection + color) consumes them unchanged. Encoding from the bbox
        # crop yields the byte-identical record a full-frame encode would.
        dets: list[dict] = []
        for i, (crop, bb, lab, sc) in enumerate(kept):
            record = encode_mask_record(crop, bbox_rcrc=bb)
            if record is None:
                continue
            dets.append(
                {
                    **record,
                    "object_id": i,
                    "label": lab,
                    "score": sc,
                }
            )
        if n_failed:
            print(
                f"[sam] WARNING: {n_failed}/{len(bounds)} SAHI tiles "
                f"FAILED (/segment returned None after retries) -- this frame is "
                f"PARTIAL-BLIND; dense-lot recall is degraded for those regions, "
                f"not a clean empty",
                flush=True,
            )
        print(
            f"[sam] SAHI native: {len(bounds)} tiles, {n_failed} failed "
            f"({len(tiledets)} tile-dets -> {len(dets)} after NMS)",
            flush=True,
        )
        return self._decode_payload_to_detections(
            {"detections": dets, "image_shape_hw": [H, W]},
            obs=obs,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_HW=(H, W),
        )

    def _segment_image_via_server(
        self,
        obs: Observation,
        concepts: list[str],
    ) -> list[DetectedObject]:
        """Stateless image-mode call to ``POST /segment``.

        Wire format mirrors ``/detect``'s detections list (same
        bbox+mask+label+score schema); the only protocol-level
        difference is that ``/segment`` carries no session and returns
        no ``frame_index``. Used by the quick-parallel sweep and any
        other disjoint-frame caller.
        """
        # Detect on obs.rgb -- the single model-input crop (native-resolution on
        # ortho, sensor render elsewhere). Masks invert to this frame's
        # shape and project via normalized ortho coords, so any input size is
        # handled by the letterbox + decode.
        rgb = obs.rgb
        assert rgb is not None

        orig_h, orig_w = int(rgb.shape[0]), int(rgb.shape[1])
        canvas, scale, pad_top, pad_left = _letterbox_to_sam(rgb)
        json_body = {
            "image_b64": base64.b64encode(_png_wire_bytes(canvas)).decode("ascii"),
            "concepts": concepts,
            "score_threshold": float(self.score_threshold),
            "min_area_px": int(self.min_area_px),
            "max_per_image": int(self.max_per_step),
            "resolution": _SAM_INPUT_SIZE,
        }
        # Stateless POST per call to defeat urllib3 stale-pool RSTs in the
        # long-running runner. Default ``requests`` Session pools sockets
        # per host; idle pool entries sit through the multi-minute compile
        # phase, the tunnel / FastAPI server reaps them (RST not FIN),
        # urllib3 can't detect half-closed state, and every POST blows up
        # with ``ConnectionReset(104)`` across all retries and all replicas
        # — a fresh process has an empty pool which is why isolated
        # reproducers always succeed. See urllib3#944, urllib3#2751,
        # urllib3#1301, requests#4937. Fix: per-call ``with
        # requests.Session()`` (empty pool), pool size 1 (nothing else in
        # the process can poison it), an explicit ``Retry`` that does NOT
        # default-exclude POST, and ``Connection: close`` so the server
        # also closes fresh.
        payload = _post_segment_avoid_stale_pool(picker=_pick_url, json_body=json_body)
        if payload is None:
            # OUTAGE, not a clean empty frame. ``None`` means the poster
            # exhausted its replica × retry budget (transport fault / 5xx) or
            # hit a client-side 4xx — the server gave NO usable answer. A
            # genuinely empty frame returns a ``{"detections": []}`` payload
            # (not ``None``), so this branch is unambiguously an outage.
            # Returning ``[]`` here would score the episode a false SR=0 that
            # is indistinguishable from a real no-detections frame; instead we
            # raise the OUTAGE SENTINEL so the flight-path caller re-queues the
            # episode (blocker V5-F2).
            raise SamServerOutage(
                "SAM /segment returned no usable response (transport fault / "
                "5xx / 4xx after retries) — whole-frame outage, not an empty "
                "detection set"
            )
        return self._decode_payload_to_detections(
            payload,
            obs=obs,
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left,
            orig_HW=(orig_h, orig_w),
        )

    def _decode_payload_to_detections(
        self,
        payload: dict,
        *,
        obs: Observation,
        scale: float,
        pad_top: int,
        pad_left: int,
        orig_HW: tuple[int, int],
    ) -> list[DetectedObject]:
        """Shared decode for the single-frame and SAHI ``/segment`` payloads.

        Inverts the letterbox, projects each mask through obs.depth,
        and packs the result into a ``DetectedObject``.
        """
        orig_h, orig_w = orig_HW
        wire_H, wire_W = payload.get(
            "image_shape_hw", [_SAM_INPUT_SIZE, _SAM_INPUT_SIZE]
        )

        # --- PASS 1: decode every mask once and collect, per class, (a) the
        # typical clean single-component instance area (sets the split noise
        # floor) and (b) a per-pixel coverage count (sets the redundancy check).
        decoded: list[tuple[int, dict, np.ndarray]] = []
        class_ref_areas: dict[str, list[int]] = {}
        coverage_by_label: dict[str, np.ndarray] = {}
        dets_wire = list(payload.get("detections", []))
        n_workers = max(1, min(int(self.decode_workers), len(dets_wire) or 1))

        def _decode_one(idx_det):
            """Per-det decode: pure (no shared writes), so dets run
            concurrently; the accumulator writes happen below on the main
            thread in det-index order (ex.map preserves it)."""
            det_idx, det = idx_det
            region = _wire_mask_region(
                det,
                wire_hw=(wire_H, wire_W),
                scale=scale,
                pad_top=pad_top,
                pad_left=pad_left,
                frame_hw=(orig_h, orig_w),
            )
            if region is None or region.area < self.min_area_px:
                return None
            comps = _connected_components(region)
            primary = comps[0] if len(comps) == 1 else max(comps, key=lambda c: c.area)
            return det_idx, det, region, comps, primary

        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="sam-decode"
        ) as ex:
            pass1 = list(ex.map(_decode_one, enumerate(dets_wire)))
        for res in pass1:
            if res is None:
                continue
            det_idx, det, region, comps, primary = res
            label = str(det["label"])
            decoded.append((det_idx, det, region))
            # Per-class PRIMARY coverage = sum of each mask's LARGEST component
            # (the instance pass 2 keeps). split_into_instances' redundancy check
            # keys off this, so a small car covered only by *merged* masks (never
            # a primary) is not silently dropped. One label pass = both this and
            # the single-component ref-area check below.
            cov = coverage_by_label.get(label)
            if cov is None:
                cov = np.zeros(region.frame_shape, dtype=np.int16)
                coverage_by_label[label] = cov
            # Through the primary's own window: a frame-sized ``cov += primary``
            # touched every pixel of the frame for every detection.
            primary.window(cov)[primary.crop] += 1
            # Reference area = clean single-component same-class masks only, so
            # the merged/speckled masks we are filtering never bias the typical
            # single-instance size.
            if len(comps) == 1:
                class_ref_areas.setdefault(label, []).append(region.area)
        class_ref: dict[str, float] = {
            lab: float(np.median(areas))
            for lab, areas in class_ref_areas.items()
            if len(areas) >= _INSTANCE_SPLIT_MIN_REF_SAMPLES
        }

        # --- PASS 2: split each mask into per-instance connected components
        # (single-component = unchanged) and pack one DetectedObject per kept
        # component. A mask whose EVERY kept component fails depth-projection
        # falls back to projecting the merged mask, so the split can never drop a
        # detection the pre-split path would have kept. Telemetry below makes the
        # split falsifiable.
        def _pack(region, det_idx, obj_id, score, label, comp_idx, n_comp):
            """Project + color + pack one region into a DetectedObject, or None
            if it fails depth-projection. Closes over obs/self.

            No frame-sized array is built anywhere on this path: the region
            travels from the wire, through the split, into the projection and
            the colour read, and out on the DetectedObject."""
            r0, c0, r1, c1 = region.bbox
            area = region.area
            # Real depth or no detection: the projection inverts the
            # yaw-rotated view-rectangle affine over ``obs.ortho_view_corners_xy``
            # and drops the mask when fewer than ``_MIN_VALID_DEPTH_PIXELS``
            # pixels have finite, positive depth.
            projected = project_mask_to_world_ortho(
                region,
                depth=obs.depth,
                view_corners_xy=obs.ortho_view_corners_xy,
                tile=obs.tile,
            )
            if projected is None:
                return None
            attrs: dict[str, str] = {}
            _set_color_attr(attrs, obs.rgb, region)
            # Distinct pre-tracker track_id per split component (the world tracker
            # re-stamps ``wt_<id>`` by world position, but keep the source id
            # unique so no pre-tracker consumer folds two split instances).
            suffix = f"_c{comp_idx}" if n_comp > 1 else ""
            track_id = f"sam3_1_img_{det_idx}{suffix}"
            metadata: dict[str, object] = {
                "concept": label,
                "sam3_1_object_id": obj_id,
                "sam3_1_score": score,
                "track_id": track_id,
                "pixel_bbox": [r0, c0, r1, c1],
                "pixel_area": area,
                "mask_pixel_count": int(projected.n_valid_pixels),
                "model_id": self.model_id,
                "frame_index": -1,
            }
            if projected.tile_cells is not None and projected.tile_cells.size:
                metadata["tile_cells"] = projected.tile_cells.astype(
                    np.int32, copy=False
                )
            return DetectedObject(
                world_xy=projected.world_xy,
                label=label,
                confidence=score,
                class_hint=resolve_class_hint(label),
                source_tier="sam3_1_image",
                metadata=metadata,
                attributes=attrs,
                mask=region,
            )

        def _split_and_pack(item):
            """Per-det split + pack: pure given the completed PASS-1
            accumulators (class_ref / coverage_by_label are read-only
            here). Returns (packs, fallback): the non-None packed
            components in comp order, and the merged-mask fallback pack
            (computed only when every component failed projection —
            fall back to the merged mask so the split never loses a
            detection the pre-split path would have kept)."""
            det_idx, det, region = item
            label = str(det["label"])
            obj_id = int(det["object_id"])
            score = float(det["score"])
            components = split_into_instances(
                region,
                ref_area_px=class_ref.get(label),
                min_area_px=self.min_area_px,
                frac=_INSTANCE_SPLIT_AREA_FRAC,
                coverage_count=coverage_by_label.get(label),
            )
            n_comp = len(components)
            packs = []
            for comp_idx, comp in enumerate(components):
                d = _pack(comp, det_idx, obj_id, score, label, comp_idx, n_comp)
                if d is not None:
                    packs.append(d)
            fallback = (
                _pack(region, det_idx, obj_id, score, label, 0, 1)
                if not packs
                else None
            )
            return packs, fallback

        # ponytail: speculative pack past the cap; gate per-det submission
        # on cap if profiles show waste.
        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="sam-pack"
        ) as ex:
            pass2 = list(ex.map(_split_and_pack, decoded))

        # Ordered replay of the serial append/cap/fallback semantics: the
        # cap truncates at the same global position, so out's contents and
        # order — and the counters — are byte-identical to the serial loop.
        out: list[DetectedObject] = []
        hit_cap = False
        n_union_fallback = 0
        n_lost = 0
        for packs, fallback in pass2:
            emitted = 0
            for d in packs:
                out.append(d)
                emitted += 1
                if len(out) >= self.max_per_step:
                    hit_cap = True
                    break
            if emitted == 0 and not hit_cap:
                if fallback is not None:
                    out.append(fallback)
                    n_union_fallback += 1
                    if len(out) >= self.max_per_step:
                        hit_cap = True
                else:
                    n_lost += 1  # no valid depth anywhere — same as pre-split
            if hit_cap:
                break
        # ``hit_cap`` gets its own trigger, not a ride on the fallback counters:
        # printed only alongside a fallback or a lost detection, a run could
        # truncate every stop at max_per_step and say nothing. A cap that
        # binds is the loudest thing this function knows.
        if n_union_fallback or n_lost or hit_cap:
            print(
                f"[sam] split: {len(decoded)} masks -> {len(out)} dets; "
                f"union_fallback={n_union_fallback} lost(no-depth)={n_lost} "
                f"cap_hit={hit_cap} max_per_step={self.max_per_step}",
                flush=True,
            )
        return out
