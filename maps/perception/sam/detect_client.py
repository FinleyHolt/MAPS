"""SAM 3.1 image adapter: per-frame POST to the segment server's ``/segment``
(``serving/sam_server.py``).

One batched forward per image runs every concept decoder. The client gets back
bbox-cropped bit-packed masks, projects them to world coords and reads colour
locally. Object identity across an episode comes from the world-space tracker.
"""

from __future__ import annotations

import base64
import itertools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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

# SAM 3.1's encoder input. The client aspect-resizes and letterbox-pads to it,
# so the wire carries ~200-400 KB instead of a multi-MB 4K PNG and the server
# sees the same pixels.
_SAM_INPUT_SIZE = 1008


class SamServerOutage(RuntimeError):
    """The SAM ``/segment`` server gave no valid response for a whole detect
    call: transport faults, 5xx, 408 or 429 until the poster's replica x retry
    budget ran out. A ``200`` with zero detections is an empty frame, never an
    outage. Raised outside the model's code (``error_taxonomy._exc_is_sam_infra``)
    it ends the episode as infra, which a resume flies again; inside a mission
    run block it is served to the model as that line's error.
    """


class SamRequestRefused(RuntimeError):
    """The segmenter refused the request this client built: a 4xx other than
    the serve's weather (408, 429). Every replica refuses the same request
    the same way, so it is raised at once and is the method's failure, never
    a :class:`SamServerOutage`.
    """


#: 4xx statuses that are the serve's weather, not a request built wrong: a
#: request timeout and a rate limit. They move the call to another replica
#: like a 5xx.
_SAM_TRANSIENT_CLIENT_STATUSES = (408, 429)


def sam_endpoints() -> list[str]:
    """The segmenter's replica base URLs, the one resolver the client and
    endpoint discovery read.

    ``SAM_VIDEO_SERVER_URL`` wins when set, otherwise ``SAM_SERVER_URL``,
    otherwise the default replica. A comma-separated value spreads stateless
    ``/segment`` calls across replicas. A set but blank value raises
    ``ValueError``: only an unset variable means the default.
    """
    name = "SAM_VIDEO_SERVER_URL" if os.environ.get("SAM_VIDEO_SERVER_URL") else "SAM_SERVER_URL"
    raw = os.environ.get(name, DEFAULT_SAM_VIDEO_SERVER_URL)
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not eps:
        raise ValueError(f"{name} is empty")
    return eps


_rr = itertools.count()


def _pick_url() -> str:
    """The next replica, round-robin: a SAHI stop fires its tiles at once and
    round-robin spreads them evenly (random would not), and the poster's
    failover steps off a dead replica rather than redrawing it.
    ``next`` on an ``itertools.count`` is atomic, so no lock."""
    eps = sam_endpoints()
    return eps[next(_rr) % len(eps)]


def _post_segment_avoid_stale_pool(*, picker, json_body) -> dict | None:
    """POST ``/segment`` on a fresh connection every call.

    A pooled socket idles through the multi-minute compile phase, the tunnel
    or server reaps it with an RST, urllib3 cannot see the half-closed state,
    and every retry on every replica then fails with ``ConnectionReset(104)``
    (urllib3#944, #2751, #1301, requests#4937). So: a per-call
    ``requests.Session`` (empty pool), a one-slot adapter nothing else can
    poison, a ``Retry`` that covers POST (urllib3 excludes it by default), and
    ``Connection: close``.

    A 4xx other than 408 or 429 raises :class:`SamRequestRefused` at once;
    transport errors, 5xx, 408 and 429 move to the next replica via
    ``picker()``. Returns the parsed JSON, or ``None`` when all three attempts
    failed.
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
                if (
                    400 <= resp.status_code < 500
                    and resp.status_code not in _SAM_TRANSIENT_CLIENT_STATUSES
                ):
                    raise SamRequestRefused(
                        f"/segment refused the request: HTTP {resp.status_code} at {url}: "
                        f"{resp.text[:200]}"
                    )
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


# Colour is read here, over the mask pixels, as a structured attribute of a
# bare-noun SAM concept ("car", not "white car"): what the object looks like,
# like its class. The agent matches the cue's colour word to it by reasoning.
# Both a palette name and the median RGB are served, so the agent can settle
# a fuzzy case (silver vs gray) the name cannot.
#
# Perceived-RGB anchors for nearest-name. The five well-sampled car colours
# (white, gray, black, red, blue) are the median perceived (SAM-mask interior)
# RGB on the CityNav birmingham/cambridge nadir orthos, n>=15 per colour over
# the LLM-labelled GT set: +10.8pp colour family and +10.4pp white/black
# lightness over ideal sRGB anchors (2-fold CV, held-out maps). The rest keep
# their ideal sRGB anchor. "silver" has no anchor: its ideal (190,192,198)
# sits above the calibrated white on this dark imagery, so true white cars
# would snap to a word CityNav cues almost never use; a silver car named
# "white" still matches a "silver" cue at 0.8 through the achromatic family.
_COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "white": (175, 183, 191), "black": (77, 86, 98), "gray": (121, 132, 145),
    "red": (135, 78, 88), "blue": (75, 101, 136), "green": (45, 130, 70),
    "yellow": (215, 205, 70), "orange": (220, 130, 45), "brown": (110, 75, 50),
    "tan": (200, 180, 145), "gold": (200, 165, 70),
}


# Colour is sampled over the mask INTERIOR: a small nadir car mask is ~1/4-1/2
# boundary pixels straddling the asphalt, which drain chroma and darken the
# median. Measured on GT cars (colour family, n~39-50 over 3 match radii):
# +14-17pp against the full mask; a value gate was inert and white balance
# negative (Hsieh 2015; Stavrothanasopoulos 2022).
_COLOR_INTERIOR_PCTL = 50
_COLOR_INTERIOR_MIN_PX = 12


def _interior_mask(region) -> "np.ndarray":
    """The region's inner pixels (distance-from-edge >= the
    ``_COLOR_INTERIOR_PCTL`` percentile) as a selection over its crop, or the
    whole crop when that would leave too few pixels (a tiny detection)."""
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


# Per-colour reliability prior on the colour confidence: chromatic car colours
# read unreliably off dark, low-contrast nadir masks (~57-80% mislabel on red
# and blue sole cues), achromatic ones reliably (~16-19%).
_COLOR_RELIABILITY_PRIOR_DEFAULT = 0.35  # chromatic seed
_COLOR_ACHROMATIC = frozenset({"white", "black", "gray", "silver"})


def _color_reliability_prior(name: str) -> float:
    return 0.82 if name in _COLOR_ACHROMATIC else _COLOR_RELIABILITY_PRIOR_DEFAULT


def _dominant_color(rgb: "np.ndarray | None", region):
    """Median colour over the region's interior -> (name, [r, g, b], conf), or
    None on missing, mismatched or empty inputs: colour never fails a detection.
    The median shrugs off windshield and shadow outliers."""
    if region is not None and not isinstance(region, MaskRegion):
        # Outside the broad except below, so a wrong caller fails loudly
        # instead of reading as colour absent.
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
            for cname, anchor in _COLOR_PALETTE.items()
        )
        name = dists[0][1]
        # Confidence: the RGB margin between the nearest anchor and the
        # runner-up, in [0, 1] (~0 between two anchors, ~1 a clean snap), times
        # the colour's reliability prior.
        d1 = float(np.sqrt(max(0.0, dists[0][0])))
        d2 = float(np.sqrt(max(0.0, dists[1][0]))) if len(dists) > 1 else d1
        margin = (d2 - d1) / (d2 + d1) if (d1 + d2) > 0.0 else 0.0
        conf = round(max(0.0, min(1.0, margin * _color_reliability_prior(name))), 3)
        return name, [int(med[0]), int(med[1]), int(med[2])], conf
    except Exception:
        return None


def _set_color_attr(attrs: dict, rgb, region) -> None:
    """``attrs['color']``, ``['color_rgb']`` and ``['color_conf']`` (strings)
    from the region's pixels, when derivable."""
    col = _dominant_color(rgb, region)
    if col is not None:
        attrs["color"] = col[0]
        attrs["color_rgb"] = f"{col[1][0]},{col[1][1]},{col[1][2]}"
        attrs["color_conf"] = f"{col[2]:.3f}"


# Per-instance split. SAM 3.x emits one mask per DETR query with no instance
# separation, NMS or component split (arXiv 2511.16719 §3), so a mask over tiny
# dense objects can span two disjoint cars or carry speckle. Each mask splits
# into connected components; the noise floor is relative to the class's
# typical single-instance area (0.10 of the median car mask ~ 1 m^2 at
# 0.25 m/px), so it needs no per-class prior.
_INSTANCE_SPLIT_AREA_FRAC = 0.10
_INSTANCE_SPLIT_MIN_REF_SAMPLES = 5
# A non-largest component more than this fraction covered by ANOTHER same-class
# detection's primary is a worse partial of that neighbour, and is dropped.
_INSTANCE_SPLIT_REDUNDANT_FRAC = 0.5


def _connected_components(region: MaskRegion) -> "list[MaskRegion]":
    """8-connected components of a region, as regions over the same frame.

    ``[region]`` itself when single-component. Components come in raster
    order (``ndimage.label`` on the crop), which matters: ``max(comps,
    key=area)`` takes the FIRST maximum.
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
    """A wire mask (a bbox crop in the letterboxed frame the server saw) as a
    region of the frame, or ``None`` when it lands empty after clipping.

    Under an identity letterbox (native SAHI tiles, the SAHI repack) the wire
    box is already a frame box and the crop is used as is. A real letterbox is
    undone on the materialised frame: a NEAREST resize does not decompose per
    box.
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
        # Clipped as decode_mask_full clips.
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
    """Split a SAM concept mask into per-instance connected components.

    A single-component mask comes back as ``[region]``. Otherwise the largest
    component is kept as the detection's primary, and each other one is
    dropped as speckle (area under ``max(min_area_px, frac * ref_area_px)``,
    ``ref_area_px`` the median clean same-class area, None with too few
    samples) or as redundant (more than ``redundant_frac`` of its pixels under
    ``coverage_count``, the per-pixel count of same-class PRIMARIES: another
    detection already owns it), and kept otherwise: a unique component is a
    car nothing else found. Keying redundancy off primaries, not raw overlap,
    keeps a small car that two merged masks both cover. Never empty.
    """
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
        # Disjoint from its own primary, so cov >= 1 is ANOTHER detection's.
        if cov is not None and float((c.window(cov)[c.crop] >= 1).mean()) > redundant_frac:
            continue  # redundant sliver of an already-detected neighbor
        kept.append(c)
    return kept


# zlib level of the wire PNG: level 1 encodes a 1008 tile in 79 ms against
# 259 ms at the default 6, for 0.4% more bytes. Every level is lossless, so SAM
# sees the same pixels. JPEG q95 moved 12% of detections against a
# bit-deterministic PNG control, so the wire stays PNG.
_PNG_WIRE_LEVEL = 1


def _png_wire_bytes(canvas: np.ndarray) -> bytes:
    """PNG-encode a letterboxed frame for the ``/segment`` wire (cv2: faster
    and smaller than PIL at the same level).

    cv2 writes BGR and the server decodes RGB with PIL, hence ``[:, :, ::-1]``;
    ``test_wire_png_is_lossless`` decodes through PIL to hold it.
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
SEGMENT_READ_TIMEOUT_S = 660


def _letterbox_to_sam(rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Aspect-preserving resize + center-pad to ``_SAM_INPUT_SIZE``:
    ``(canvas, scale, pad_top, pad_left)`` with ``letterboxed = scale * orig +
    pad``. A native 1008 tile passes through unresampled.
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
        return np.ascontiguousarray(rgb, dtype=np.uint8), 1.0, 0, 0
    scale = float(_SAM_INPUT_SIZE) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray(rgb.astype(np.uint8)).resize(
            # LANCZOS: Pillow scales the support to the factor, so a shrink
            # is anti-aliased.
            (new_w, new_h), resample=Image.Resampling.LANCZOS
        )
    )
    canvas = np.zeros((_SAM_INPUT_SIZE, _SAM_INPUT_SIZE, 3), dtype=np.uint8)
    pad_top = (_SAM_INPUT_SIZE - new_h) // 2
    pad_left = (_SAM_INPUT_SIZE - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, pad_top, pad_left


#: Threads for the per-detection mask decode, projection and colour read;
#: output order is the serial order at any width.
_DECODE_WORKERS = 4


@dataclass
class SamDetectClient:
    """Adapter onto the SAM 3.1 image server (stateless ``/segment``).

    Every call ships the full concept list: the encoder runs once and every
    concept decoder runs in the same forward.
    """

    concept_provider: ReasonerUnionConceptProvider
    model_id: str = "facebook/sam3.1"
    score_threshold: float = 0.3
    min_area_px: int = 16
    max_per_step: int = 64
    # Force native-resolution SAHI tiling on a frame that already fits the
    # encoder (a larger frame always tiles): 1008 px tiles fed 1:1, remapped
    # to the frame, cross-tile IoU-NMS dedup. Recovers dense-lot car recall
    # full-frame SAM loses to sub-pixel cars (~67% -> 82% measured).
    sahi: bool = False
    sahi_overlap: float = 0.2
    sahi_nms_iou: float = 0.5
    # SAHI tiles whose ``/segment`` gave no usable response after retries,
    # over this client's life (one episode; the row's ``sam_tiles_failed``).
    # A frame with some failed tiles is partial-blind, not empty.
    sahi_tiles_failed: int = field(default=0, init=False)
    _failed_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def detect(
        self,
        obs: Observation,
        pose_xy: tuple[float, float],
        instruction_hint: str | None = None,
    ) -> list[DetectedObject]:
        """Detect the provider's concepts on one frame; ``[]`` with no
        concepts or no picture."""
        del pose_xy, instruction_hint  # obs.ortho_view_corners_xy drives the projection
        if obs.rgb is None:
            return []
        if obs.ortho_view_corners_xy is None:
            raise ValueError(
                "SamDetectClient.detect requires obs.ortho_view_corners_xy "
                "(the view square a mask projects through)"
            )
        concepts = list(self.concept_provider.concepts)
        return self._segment(obs, concepts) if concepts else []

    def _segment(
        self, obs: Observation, concepts: list[str]
    ) -> list[DetectedObject]:
        """SAHI tiles when the frame exceeds the encoder input in either
        dimension (a letterbox shrink would hand the encoder a fraction of its
        pixels) or ``sahi`` forces it; else one full-frame ``/segment``."""
        h, w = obs.rgb.shape[:2]
        if self.sahi or h > _SAM_INPUT_SIZE or w > _SAM_INPUT_SIZE:
            return self._sahi_segment(obs, concepts)
        return self._segment_image_via_server(obs, concepts)

    def _post_frame(
        self, rgb: np.ndarray, concepts: list[str], *, image_first: bool = False
    ) -> tuple[dict | None, float, int, int]:
        """Letterbox ``rgb`` and POST it to ``/segment``: ``(payload, scale,
        pad_top, pad_left)``, the payload ``None`` when the poster gave up.
        ``image_first`` puts ``image_b64`` ahead of the knobs in the body."""
        canvas, scale, pad_top, pad_left = _letterbox_to_sam(rgb)
        image = {"image_b64": base64.b64encode(_png_wire_bytes(canvas)).decode("ascii")}
        knobs = {
            "concepts": concepts,
            "score_threshold": float(self.score_threshold),
            "min_area_px": int(self.min_area_px),
            "max_per_image": int(self.max_per_step),
            "resolution": _SAM_INPUT_SIZE,
        }
        body = {**image, **knobs} if image_first else {**knobs, **image}
        payload = _post_segment_avoid_stale_pool(picker=_pick_url, json_body=body)
        return payload, scale, pad_top, pad_left

    @staticmethod
    def _tile_bounds(
        H: int, W: int, overlap: float
    ) -> "list[tuple[int, int, int, int]]":
        """Fixed native-size tiling: every tile is EXACTLY ``_SAM_INPUT_SIZE`` so
        it is fed to SAM 1:1. The tile count follows from the frame size and
        overlap: a 2016 render gives a 3x3 grid (starts {0, 504, 1008})."""
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
        only. A ``None`` crop is an empty mask carried for the tile-det count
        and dropped here. This is the cross-tile seam dedup; the world-space
        tracker dedups again downstream."""
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
        """Native-resolution slicing-aided detection: ``/segment`` each
        ``_SAM_INPUT_SIZE`` tile at 1:1, carry each mask as its bbox crop in
        frame coordinates, cross-tile IoU-NMS dedup, then re-pack into one
        frame-sized wire payload for the same decode as the single-frame call.
        """
        rgb = obs.rgb
        assert rgb is not None
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        bounds = self._tile_bounds(H, W, self.sahi_overlap)

        def _one_tile(bound) -> list[tuple] | None:
            """One tile's ``(crop, bbox_rcrc, area, label, score)`` items in
            wire order (crop ``None`` for an empty mask), or ``None`` when its
            ``/segment`` failed. Reads only shared-immutable state."""
            r0, r1, c0, c1 = bound
            sub = rgb[r0:r1, c0:c1]
            payload, scale, pad_top, pad_left = self._post_frame(sub, concepts)
            if payload is None:
                return None
            wire_hw = payload.get("image_shape_hw", [_SAM_INPUT_SIZE, _SAM_INPUT_SIZE])
            tile_out: list[tuple] = []
            for det in payload.get("detections", []):
                region = _wire_mask_region(
                    det,
                    wire_hw=wire_hw,
                    scale=scale,
                    pad_top=pad_top,
                    pad_left=pad_left,
                    frame_hw=sub.shape[:2],
                )
                area = 0 if region is None else region.area
                if area < self.min_area_px:
                    continue
                lab, sc = str(det["label"]), float(det["score"])
                if region is None:
                    tile_out.append((None, (0, 0, 0, 0), 0, lab, sc))
                    continue
                tight = region.tightened()
                lr0, lc0, lr1, lc1 = tight.bbox
                tile_out.append((tight.crop.copy(), (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1), area, lab, sc))
            return tile_out

        # One thread per tile: width costs threads, not GPU (the servers'
        # SAM_MAX_INFLIGHT gates the work), and a narrower pool leaves a tail
        # wave. ex.map keeps bounds order, so NMS tie-breaking and the repack's
        # object_ids are the serial loop's.
        with ThreadPoolExecutor(max_workers=len(bounds), thread_name_prefix="sahi-tile") as ex:
            per_tile = list(ex.map(_one_tile, bounds))
        tiledets = [item for tile_out in per_tile if tile_out is not None for item in tile_out]
        n_failed = sum(tile_out is None for tile_out in per_tile)
        if n_failed:
            with self._failed_lock:
                self.sahi_tiles_failed += n_failed
        if n_failed == len(bounds):
            # Every tile failed: a whole-frame outage, not an empty frame (a
            # 200 with zero detections is not a failed tile).
            raise SamServerOutage(
                f"SAM SAHI outage: all {len(bounds)} tiles failed "
                "(/segment returned no usable response after retries) — "
                "whole-frame outage, not a partial or empty frame"
            )
        kept = self._nms_masks(tiledets, self.sahi_nms_iou)
        # The merged masks as a wire payload whose wire frame IS the frame
        # (identity letterbox).
        dets: list[dict] = []
        for i, (crop, bb, lab, sc) in enumerate(kept):
            record = encode_mask_record(crop, bbox_rcrc=bb)
            if record is None:
                continue
            dets.append({**record, "object_id": i, "label": lab, "score": sc})
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
        """One full-frame ``POST /segment``, letterboxed, decoded back to the
        frame."""
        rgb = obs.rgb
        assert rgb is not None
        payload, scale, pad_top, pad_left = self._post_frame(rgb, concepts, image_first=True)
        if payload is None:
            raise SamServerOutage(
                "SAM /segment returned no usable response (transport fault / "
                "5xx / 408 / 429 after retries) — whole-frame outage, not an "
                "empty detection set"
            )
        return self._decode_payload_to_detections(
            payload,
            obs=obs,
            scale=scale,
            pad_top=pad_top,
            pad_left=pad_left,
            orig_HW=(int(rgb.shape[0]), int(rgb.shape[1])),
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
        """Decode a ``/segment`` payload (single-frame or SAHI repack): undo
        the letterbox, split each mask into instances, project each through
        ``obs.depth`` and pack it as a ``DetectedObject``, in wire order on
        thread pools.
        """
        orig_h, orig_w = orig_HW
        wire_H, wire_W = payload.get(
            "image_shape_hw", [_SAM_INPUT_SIZE, _SAM_INPUT_SIZE]
        )

        # Pass 1: decode every mask and collect, per class, the clean
        # single-component area (the split's noise floor) and a per-pixel count
        # of primaries (the redundancy check).
        decoded: list[tuple[int, dict, np.ndarray]] = []
        class_ref_areas: dict[str, list[int]] = {}
        coverage_by_label: dict[str, np.ndarray] = {}
        dets_wire = list(payload.get("detections", []))
        n_workers = max(1, min(_DECODE_WORKERS, len(dets_wire) or 1))

        def _decode_one(idx_det):
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
            # Coverage counts primaries (each mask's largest component), so a
            # small car covered only by merged masks survives the split.
            cov = coverage_by_label.get(label)
            if cov is None:
                cov = np.zeros(region.frame_shape, dtype=np.int16)
                coverage_by_label[label] = cov
            primary.window(cov)[primary.crop] += 1
            if len(comps) == 1:
                class_ref_areas.setdefault(label, []).append(region.area)
        class_ref: dict[str, float] = {
            lab: float(np.median(areas))
            for lab, areas in class_ref_areas.items()
            if len(areas) >= _INSTANCE_SPLIT_MIN_REF_SAMPLES
        }

        # Pass 2: one DetectedObject per kept component; a mask whose every
        # component fails projection falls back to the merged mask.
        def _pack(region, det_idx, obj_id, score, label, comp_idx, n_comp):
            """One region projected, coloured and packed, or None when fewer
            than ``_MIN_VALID_DEPTH_PIXELS`` of its pixels have real depth."""
            r0, c0, r1, c1 = region.bbox
            area = region.area
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
            # Unique per split component until the world tracker re-stamps it.
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
            """``(packs, fallback)``: the packed components in order, and the
            merged-mask pack when none of them projected."""
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

        out: list[DetectedObject] = []
        hit_cap = False
        n_union_fallback = 0
        n_lost = 0
        for packs, fallback in pass2:
            if not packs:
                if fallback is None:
                    n_lost += 1  # no valid depth anywhere
                    continue
                packs = [fallback]
                n_union_fallback += 1
            for d in packs:
                out.append(d)
                if len(out) >= self.max_per_step:
                    hit_cap = True
                    break
            if hit_cap:
                break
        # A binding cap prints on its own: a run truncating every stop must say so.
        if n_union_fallback or n_lost or hit_cap:
            print(
                f"[sam] split: {len(decoded)} masks -> {len(out)} dets; "
                f"union_fallback={n_union_fallback} lost(no-depth)={n_lost} "
                f"cap_hit={hit_cap} max_per_step={self.max_per_step}",
                flush=True,
            )
        return out
