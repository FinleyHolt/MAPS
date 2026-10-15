"""SAM 3.1 client: each frame goes to ``/segment`` (``serving/sam_server.py``)
as native tiles; the masks come back, are deduped across tiles, projected to
world coordinates and read for colour.
"""

from __future__ import annotations

import base64
import itertools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from maps.perception.base import DetectedObject
from maps.perception.geometry.geolocate import (
    project_mask_to_world_ortho,
)
from maps.perception.observation import Observation
from maps.util.mask_codec import decode_mask_crop
from maps.util.mask_region import MaskRegion

DEFAULT_SAM_SERVER_URL = "http://127.0.0.1:8772"

# SAM 3.1's encoder input (the serve's ``sam_engine.RESOLUTION``): every tile
# is this square, fed 1:1.
_SAM_INPUT_SIZE = 1008

#: The segmenter's options the client sets: the speckle floor the server
#: applies and the client again after the split, the per-frame detection
#: backstop (no per-concept cap, so every same-class instance stays
#: countable), and the tile overlap and cross-tile IoU-NMS threshold. The
#: score floor is the serve's own (``sam_engine.SCORE_THRESHOLD``).
MIN_AREA_PX = 32
MAX_PER_STEP = 1024
SAHI_OVERLAP = 0.2
SAHI_NMS_IOU = 0.5


class SamServerOutage(RuntimeError):
    """No tile of a frame got a usable ``/segment`` response
    (:func:`_post_segment` gave up on each); a ``200`` with zero detections is
    an empty frame. Outside the model's code it ends the episode as infra;
    inside a run block it is served as that line's error."""


class SamRequestRefused(RuntimeError):
    """The segmenter refused the request this client built (a 4xx outside
    ``_SAM_TRANSIENT_CLIENT_STATUSES``): every replica would, so it is raised
    at once, never an outage."""


#: 4xx statuses that are the serve's weather, not a request built wrong: a
#: request timeout and a rate limit. They move the call to another replica
#: like a 5xx. Kept apart from ``maps.llm.http.TRANSIENT_CLIENT_STATUS``,
#: whose 402 is a hosted reasoner's quota check: the self-hosted segmenter's
#: 402 is a refusal.
_SAM_TRANSIENT_CLIENT_STATUSES = (408, 429)


def sam_endpoints() -> list[str]:
    """The segmenter's replica base URLs from ``SAM_SERVER_URL`` (a comma
    list spreads calls across replicas), the one resolver the client and
    endpoint discovery read. Unset is the default replica; set but blank,
    or unset beside the flown station's ``SAM_VIDEO_SERVER_URL``, raises
    ``ValueError``."""
    if "SAM_SERVER_URL" not in os.environ and os.environ.get("SAM_VIDEO_SERVER_URL"):
        raise ValueError("SAM_VIDEO_SERVER_URL is not read; set SAM_SERVER_URL")
    raw = os.environ.get("SAM_SERVER_URL", DEFAULT_SAM_SERVER_URL)
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not eps:
        raise ValueError("SAM_SERVER_URL is empty")
    return eps


_rr = itertools.count()


def _pick_url(eps: list[str]) -> str:
    """The next of ``eps``, round-robin, so a frame's tiles spread evenly and
    a retry steps off the replica that failed (``next`` on a count is atomic)."""
    return eps[next(_rr) % len(eps)]


def _post_segment(json_body: dict) -> dict | None:
    """POST ``/segment``, the one retry layer: the parsed JSON, or ``None``.

    A 4xx outside ``_SAM_TRANSIENT_CLIENT_STATUSES`` raises
    :class:`SamRequestRefused`. A connect error, one of those statuses or a
    5xx goes to the next replica after
    ``SEGMENT_BACKOFF_S``. A read timeout is never resent to the replica that
    timed out (the server still holds the request): it moves to another, or
    fails when none is left. ``SEGMENT_DEADLINE_S`` bounds the whole. Each
    attempt opens a fresh connection: a pooled socket the tunnel reaped
    during a long compile reads as half-open (urllib3#944, #2751).
    """
    import requests

    deadline = time.monotonic() + SEGMENT_DEADLINE_S
    eps = sam_endpoints()
    last: Exception | str | None = None
    for attempt in range(len(SEGMENT_BACKOFF_S) + 1):
        left = deadline - time.monotonic()
        if not eps or left <= 0.0:
            break
        url = _pick_url(eps)
        try:
            with requests.Session() as s:
                resp = s.post(url + "/segment", json=json_body, headers={"Connection": "close"},
                              timeout=(min(SEGMENT_CONNECT_TIMEOUT_S, left), min(SEGMENT_READ_TIMEOUT_S, left)))
                payload = resp.json() if resp.ok else None
        except requests.exceptions.ReadTimeout as e:
            last = e
            eps = [u for u in eps if u != url]
            continue
        except requests.exceptions.RequestException as e:
            last = e
        else:
            if resp.ok:
                return payload
            if resp.status_code < 500 and resp.status_code not in _SAM_TRANSIENT_CLIENT_STATUSES:
                raise SamRequestRefused(
                    f"/segment refused the request: HTTP {resp.status_code} at {url}: {resp.text[:200]}"
                )
            last = f"HTTP {resp.status_code} at {url}"
        if attempt < len(SEGMENT_BACKOFF_S):
            backoff = SEGMENT_BACKOFF_S[attempt]
            if time.monotonic() + backoff >= deadline:
                break
            time.sleep(backoff)
    print(f"[sam] /segment gave no usable response: {last}", flush=True)
    return None


# Colour is an attribute of a bare-noun concept ("car", not "white car"),
# served as the nearest palette name and the median RGB. White, gray, black,
# red and blue are the median SAM-mask interiors on the CityNav nadir orthos,
# the rest ideal sRGB.
_COLOR_PALETTE: dict[str, tuple[int, int, int]] = {
    "white": (175, 183, 191), "black": (77, 86, 98), "gray": (121, 132, 145),
    "red": (135, 78, 88), "blue": (75, 101, 136), "green": (45, 130, 70),
    "yellow": (215, 205, 70), "orange": (220, 130, 45), "brown": (110, 75, 50),
    "tan": (200, 180, 145), "gold": (200, 165, 70),
}


# Colour is sampled over the mask INTERIOR: a small nadir car mask is largely
# boundary pixels straddling the asphalt, which drain chroma and darken the
# median.
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
# read unreliably off dark, low-contrast nadir masks, achromatic ones reliably.
_COLOR_RELIABILITY_PRIOR_DEFAULT = 0.35  # chromatic seed
_COLOR_RELIABILITY_PRIOR_ACHROMATIC = 0.82
_COLOR_ACHROMATIC = frozenset({"white", "black", "gray"})


def _dominant_color(rgb: np.ndarray, region: MaskRegion) -> tuple[str, list[int], float]:
    """Median colour over the region's interior of its frame ``rgb``: ``(name,
    [r, g, b], conf)``. The median shrugs off windshield and shadow outliers."""
    px = region.window(np.asarray(rgb))[_interior_mask(region)][:, :3].astype(np.float32)
    med = np.median(px, axis=0)
    dists = sorted(
        (float(np.sum((med - np.asarray(anchor, dtype=np.float32)) ** 2)), cname)
        for cname, anchor in _COLOR_PALETTE.items()
    )
    name = dists[0][1]
    # Confidence: the RGB margin between the nearest anchor and the
    # runner-up, in [0, 1] (~0 between two anchors, ~1 a clean snap), times
    # the colour's reliability prior.
    d1, d2 = float(np.sqrt(dists[0][0])), float(np.sqrt(dists[1][0]))
    margin = (d2 - d1) / (d2 + d1) if (d1 + d2) > 0.0 else 0.0
    prior = _COLOR_RELIABILITY_PRIOR_ACHROMATIC if name in _COLOR_ACHROMATIC else _COLOR_RELIABILITY_PRIOR_DEFAULT
    conf = round(max(0.0, min(1.0, margin * prior)), 3)
    return name, [int(med[0]), int(med[1]), int(med[2])], conf


# Per-instance split: SAM 3.x emits one mask per query with no instance
# separation (arXiv 2511.16719 §3), so a mask can span two cars or carry
# speckle; the noise floor is a share of the class's median single area.
_INSTANCE_SPLIT_AREA_FRAC = 0.10
_INSTANCE_SPLIT_MIN_REF_SAMPLES = 5
# A non-largest component more than this fraction covered by ANOTHER same-class
# detection's primary is a worse partial of that neighbour, and is dropped.
_INSTANCE_SPLIT_REDUNDANT_FRAC = 0.5


def _connected_components(region: MaskRegion) -> "list[MaskRegion]":
    """8-connected components of a region, each re-boxed, in raster order
    (``max(comps, key=area)`` takes the FIRST maximum); ``[region]`` when single."""
    from scipy import ndimage

    labels, n = ndimage.label(region.crop, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return [region]
    return [region.with_crop(labels == i).tightened() for i in range(1, n + 1)]


def record_region(record: dict, frame_hw: tuple[int, int]) -> "MaskRegion | None":
    """A wire mask record as a region of a ``frame_hw`` frame, clipped to it;
    ``None`` when nothing is left."""
    frame_h, frame_w = frame_hw
    r0, c0, r1, c1 = (int(v) for v in record["pixel_bbox_rcrc"])
    r1 = min(r1, frame_h - 1)
    c1 = min(c1, frame_w - 1)
    if r1 < r0 or c1 < c0:
        return None
    crop = decode_mask_crop(record)[: r1 - r0 + 1, : c1 - c0 + 1]
    if not crop.any():
        return None
    return MaskRegion(crop, (r0, c0, r1, c1), (frame_h, frame_w))


def split_into_instances(
    comps: "list[MaskRegion]",
    *,
    ref_area_px: float | None = None,
    coverage_count: "np.ndarray | None" = None,
) -> "list[MaskRegion]":
    """A SAM concept mask's connected components (:func:`_connected_components`)
    as per-instance components, never empty: the largest is kept, each other one dropped as speckle (under ``MIN_AREA_PX``
    or ``_INSTANCE_SPLIT_AREA_FRAC`` of ``ref_area_px``, the median clean
    same-class area) or as redundant (over ``_INSTANCE_SPLIT_REDUNDANT_FRAC``
    under ``coverage_count``, the per-pixel count of same-class primaries).
    """
    if len(comps) <= 1:
        return comps
    floor = MIN_AREA_PX
    if ref_area_px is not None:
        floor = max(floor, int(round(_INSTANCE_SPLIT_AREA_FRAC * float(ref_area_px))))
    order = sorted(comps, key=lambda c: c.area, reverse=True)
    kept = [order[0]]  # largest = the detection's primary instance, always kept
    for c in order[1:]:
        if c.area < floor:
            continue  # speckle
        # Disjoint from its own primary, so a count >= 1 is ANOTHER detection's.
        if coverage_count is not None and float((c.window(coverage_count)[c.crop] >= 1).mean()) > _INSTANCE_SPLIT_REDUNDANT_FRAC:
            continue  # redundant sliver of an already-detected neighbor
        kept.append(c)
    return kept


# zlib level of the wire PNG: every level is lossless, so SAM sees the same
# pixels, and level 1 is the fastest. The wire stays PNG: JPEG moves detections.
_PNG_WIRE_LEVEL = 1


def _png_wire_bytes(canvas: np.ndarray) -> bytes:
    """PNG-encode a tile for the ``/segment`` wire. cv2 writes BGR and the
    server decodes RGB with PIL, hence ``[:, :, ::-1]``."""
    import cv2

    ok, buf = cv2.imencode(
        ".png", canvas[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, _PNG_WIRE_LEVEL]
    )
    if not ok:
        raise RuntimeError("cv2 refused to PNG-encode the SAM wire frame")
    return buf.tobytes()


#: How long one /segment attempt waits to connect: past a radio dropout.
SEGMENT_CONNECT_TIMEOUT_S = 45
#: How long one /segment attempt waits for the answer: past the server's
#: ``_INFLIGHT_WAIT_S`` (600 s) admission wait, so a queued request is not abandoned.
SEGMENT_READ_TIMEOUT_S = 660
#: The pause before each retry: 21 s across the three, so the ladder spans a
#: radio dropout instead of spending every retry inside one.
SEGMENT_BACKOFF_S = (3.0, 6.0, 12.0)
#: The most one tile's /segment takes, every attempt and pause included: one
#: full server wait plus a connect ladder.
SEGMENT_DEADLINE_S = 900.0


#: Threads for the per-detection mask decode, projection and colour read;
#: output order is the serial order at any width.
_DECODE_WORKERS = 4


@dataclass
class SamDetectClient:
    """Adapter onto the SAM 3.1 image server: every ``/segment`` call ships
    the full concept list, decoded in one forward.

    ``concepts`` is the list written last through :meth:`set_concepts`, a
    wholesale replacement the ``PerceptionDirector`` pushes: the watchlist
    (the mission's declared classes, in declared order) plus the classes of
    the boosts riding a flight. Empty is valid and the common state before the model declares anything:
    ``detect`` then returns ``[]``."""

    concepts: list[str] = field(default_factory=list)
    # SAHI tiles whose ``/segment`` gave no usable response after retries,
    # over this client's life (one episode; the row's ``sam_tiles_failed``).
    # A frame with some failed tiles is partial-blind, not empty.
    sahi_tiles_failed: int = field(default=0, init=False)
    _failed_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def set_concepts(self, concepts: Iterable[str]) -> None:
        """Replace the list: case-insensitive dedup in first-seen order,
        entries stripped, blanks dropped."""
        seen: set[str] = set()
        out: list[str] = []
        for s in concepts or []:
            key = (s or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
        self.concepts = out

    def detect(self, obs: Observation) -> list[DetectedObject]:
        """Detect ``concepts`` on one frame; ``[]`` with no
        concepts or no picture."""
        if obs.rgb is None:
            return []
        if obs.ortho_view_corners_xy is None:
            raise ValueError(
                "SamDetectClient.detect requires obs.ortho_view_corners_xy "
                "(the view square a mask projects through)"
            )
        concepts = list(self.concepts)
        return self.segment(obs, concepts) if concepts else []

    @staticmethod
    def _tile_bounds(H: int, W: int) -> "list[tuple[int, int, int, int]]":
        """``(r0, r1, c0, c1)`` of the ``_SAM_INPUT_SIZE`` tiles covering an
        ``H x W`` frame, overlapping by at least ``SAHI_OVERLAP``: a 2016 px
        side gives starts {0, 504, 1008}."""
        t = _SAM_INPUT_SIZE

        def starts(size: int) -> "list[int]":
            if size <= t:
                return [0]
            step = max(1, int(round(t * (1.0 - SAHI_OVERLAP))))
            n = max(2, -(-(size - t) // step) + 1)  # ceil((size-t)/step) + 1
            return [int(round(i * (size - t) / (n - 1))) for i in range(n)]

        rs = starts(H)
        cs = starts(W)
        return [(r, min(r + t, H), c, min(c + t, W)) for r in rs for c in cs]

    @staticmethod
    def _nms_masks(
        items: "list[tuple[MaskRegion, str, float]]",
    ) -> "list[tuple[MaskRegion, str, float]]":
        """The cross-tile seam dedup: greedy IoU-NMS over ``(region, label,
        score)``, highest score first (ties in input order), pixels compared on
        the box intersection only."""
        kept: list[tuple[MaskRegion, int, str, float]] = []
        for region, lab, sc in sorted(items, key=lambda it: -it[2]):
            bb, a = region.bbox, region.area
            for k, ka, _, _ in kept:
                kbb = k.bbox
                if bb[2] < kbb[0] or bb[0] > kbb[2] or bb[3] < kbb[1] or bb[1] > kbb[3]:
                    continue  # bboxes disjoint -> no overlap
                wr0, wr1 = max(bb[0], kbb[0]), min(bb[2], kbb[2])
                wc0, wc1 = max(bb[1], kbb[1]), min(bb[3], kbb[3])
                inter = int(
                    (
                        region.crop[wr0 - bb[0] : wr1 - bb[0] + 1, wc0 - bb[1] : wc1 - bb[1] + 1]
                        & k.crop[wr0 - kbb[0] : wr1 - kbb[0] + 1, wc0 - kbb[1] : wc1 - kbb[1] + 1]
                    ).sum()
                )
                if inter and inter / (a + ka - inter) > SAHI_NMS_IOU:
                    break
            else:
                kept.append((region, a, lab, sc))
        return [(region, lab, sc) for region, _a, lab, sc in kept]

    def segment(
        self, obs: Observation, concepts: list[str]
    ) -> list[DetectedObject]:
        """Slicing-aided detection: ``/segment`` each ``_SAM_INPUT_SIZE`` tile
        1:1 (a smaller frame zero-padded bottom and right, its masks clipped),
        place the masks in the frame, dedup across tiles, decode the kept."""
        rgb = obs.rgb
        assert rgb is not None
        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        canvas = np.pad(rgb, ((0, max(0, _SAM_INPUT_SIZE - H)), (0, max(0, _SAM_INPUT_SIZE - W)), (0, 0)))
        bounds = self._tile_bounds(canvas.shape[0], canvas.shape[1])

        def _one_tile(bound) -> "list[tuple[MaskRegion, str, float]] | None":
            """One tile's ``(region, label, score)`` items, or ``None`` when its ``/segment`` failed."""
            r0, r1, c0, c1 = bound
            tile = np.ascontiguousarray(canvas[r0:r1, c0:c1], dtype=np.uint8)
            payload = _post_segment({
                "concepts": concepts,
                "min_area_px": MIN_AREA_PX,
                "max_per_image": MAX_PER_STEP,
                "image_b64": base64.b64encode(_png_wire_bytes(tile)).decode("ascii"),
            })
            if payload is None:
                return None
            tile_out = []
            for det in payload.get("detections", []):
                region = record_region(det, (min(r1, H) - r0, min(c1, W) - c0))
                if region is None or region.area < MIN_AREA_PX:
                    continue
                tight = region.tightened()
                lr0, lc0, lr1, lc1 = tight.bbox
                tile_out.append((
                    MaskRegion(tight.crop, (r0 + lr0, c0 + lc0, r0 + lr1, c0 + lc1), (H, W)),
                    str(det["label"]),
                    float(det["score"]),
                ))
            return tile_out

        # One thread per tile (the servers' SAM_MAX_INFLIGHT gates the GPU);
        # ex.map keeps bounds order, so NMS ties and object ids are serial's.
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
        kept = self._nms_masks(tiledets)
        # A failed tile is ground this frame did not see: partial-blind, not a clean empty.
        print(
            f"[sam] SAHI native: {len(bounds)} tiles, {n_failed} failed "
            f"({len(tiledets)} tile-dets -> {len(kept)} after NMS)"
            + (" -- WARNING: this frame is PARTIAL-BLIND" if n_failed else ""),
            flush=True,
        )
        return self.detections(obs, [(region, lab, i, sc) for i, (region, lab, sc) in enumerate(kept)])

    def detections(
        self,
        obs: Observation,
        masks: "list[tuple[MaskRegion | None, str, int, float]]",
    ) -> list[DetectedObject]:
        """``(region, label, object_id, score)`` masks as detections, in input
        order: each split into instances, projected through ``obs.depth`` and
        packed. A ``None`` or sub-``MIN_AREA_PX`` region is skipped."""
        # Pass 1: split every mask; per class, the clean single-component areas
        # (the noise floor) and a per-pixel count of primaries (redundancy).
        decoded: list[tuple[int, MaskRegion, list[MaskRegion], str, int, float]] = []
        class_ref_areas: dict[str, list[int]] = {}
        coverage_by_label: dict[str, np.ndarray] = {}
        n_workers = min(_DECODE_WORKERS, len(masks)) or 1

        def _decode_one(idx_mask):
            det_idx, (region, label, obj_id, score) = idx_mask
            if region is None or region.area < MIN_AREA_PX:
                return None
            comps = _connected_components(region)
            return (det_idx, region, comps, str(label), int(obj_id), float(score)), max(comps, key=lambda c: c.area)

        with ThreadPoolExecutor(
            max_workers=n_workers, thread_name_prefix="sam-decode"
        ) as ex:
            pass1 = list(ex.map(_decode_one, enumerate(masks)))
        for res in pass1:
            if res is None:
                continue
            item, primary = res
            region, comps, label = item[1], item[2], item[3]
            decoded.append(item)
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
            """One region projected, coloured and packed, or None when too few
            of its pixels carry real depth (``geolocate.project_mask_to_world_ortho``)."""
            world_xy = project_mask_to_world_ortho(
                region, depth=obs.depth, view_corners_xy=obs.ortho_view_corners_xy
            )
            if world_xy is None:
                return None
            color, med, conf = _dominant_color(obs.rgb, region)
            attrs = {"color": color, "color_rgb": f"{med[0]},{med[1]},{med[2]}", "color_conf": f"{conf:.3f}"}
            # Unique per split component until the world tracker re-stamps it.
            suffix = f"_c{comp_idx}" if n_comp > 1 else ""
            track_id = f"sam3_1_img_{det_idx}{suffix}"
            metadata: dict[str, object] = {
                "sam3_1_object_id": obj_id,
                "sam3_1_score": score,
                "track_id": track_id,
            }
            return DetectedObject(
                world_xy=world_xy,
                label=label,
                confidence=score,
                source_tier="sam3_1_image",
                metadata=metadata,
                attributes=attrs,
                mask=region,
            )

        def _split_and_pack(item):
            """``(packs, fallback)``: the packed components in order, and the
            merged-mask pack when none of them projected."""
            det_idx, region, comps, label, obj_id, score = item
            components = split_into_instances(
                comps,
                ref_area_px=class_ref.get(label),
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
        n_union_fallback = n_lost = 0
        for packs, fallback in pass2:
            if not packs:
                if fallback is None:
                    n_lost += 1  # no valid depth anywhere
                    continue
                packs = [fallback]
                n_union_fallback += 1
            out.extend(packs)
        hit_cap = len(out) > MAX_PER_STEP
        out = out[:MAX_PER_STEP]
        # A binding cap prints on its own: a run truncating every stop must say so.
        if n_union_fallback or n_lost or hit_cap:
            print(
                f"[sam] split: {len(decoded)} masks -> {len(out)} dets; "
                f"union_fallback={n_union_fallback} lost(no-depth)={n_lost} "
                f"cap_hit={hit_cap} max_per_step={MAX_PER_STEP}",
                flush=True,
            )
        return out
