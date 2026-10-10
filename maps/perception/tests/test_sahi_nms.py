"""SAHI cross-tile NMS — bbox-crop pipeline is bit-identical to full-frame.

``_sahi_segment`` carries each tile detection as its bbox crop plus full-frame
bbox instead of materializing a frame-sized mask per detection (766 × 2016²
bool allocations measured per frame). These tests pin the contract that made
that refactor safe: the kept set, its order, and the encoded wire records are
byte-identical to the full-frame reference pipeline, which lives HERE as the
reference implementation.
"""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
import pytest
from PIL import Image

from maps.perception.observation import Observation
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import (
    SamDetectClient,
    _letterbox_to_sam,
    _png_wire_bytes,
)
from maps.util.mask_codec import decode_mask_full, encode_mask_record


# ── Reference: the full-frame pipeline (previous shipped semantics) ──


def _reference_nms_full_masks(items, iou_thr):
    """Greedy IoU-NMS over FULL-FRAME masks — the reference semantics."""
    order = sorted(range(len(items)), key=lambda i: -items[i][2])
    kept = []  # (mask, bbox, area, label, score)
    for i in order:
        m, lab, sc = items[i]
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        bb = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
        a = int(m.sum())
        dup = False
        for km, kbb, ka, _, _ in kept:
            if bb[2] < kbb[0] or bb[0] > kbb[2] or bb[3] < kbb[1] or bb[1] > kbb[3]:
                continue
            inter = int((m & km).sum())
            if inter and inter / (a + ka - inter) > iou_thr:
                dup = True
                break
        if not dup:
            kept.append((m, bb, a, lab, sc))
    return [(m, lab, sc) for (m, _bb, _a, lab, sc) in kept]


def _items_from_full(full_masks):
    """Build the crop-item tuples ``_nms_masks`` consumes from full masks."""
    items = []
    for m, lab, sc in full_masks:
        rows = m.any(axis=1)
        if not rows.any():
            items.append((None, (0, 0, 0, 0), 0, lab, sc))
            continue
        cols = m.any(axis=0)
        r0 = int(rows.argmax())
        r1 = int(m.shape[0] - 1 - rows[::-1].argmax())
        c0 = int(cols.argmax())
        c1 = int(m.shape[1] - 1 - cols[::-1].argmax())
        items.append(
            (
                m[r0 : r1 + 1, c0 : c1 + 1].copy(),
                (r0, c0, r1, c1),
                int(m.sum()),
                lab,
                sc,
            )
        )
    return items


def _random_blobs(rng, n, H, W):
    """Random rectangular-ish blobs with overlapping clusters and score ties."""
    out = []
    for i in range(n):
        h = int(rng.integers(3, 40))
        w = int(rng.integers(3, 40))
        r = int(rng.integers(0, H - h))
        c = int(rng.integers(0, W - w))
        m = np.zeros((H, W), dtype=bool)
        m[r : r + h, c : c + w] = True
        # carve a corner so masks are not pure rectangles
        m[r : r + max(1, h // 3), c : c + max(1, w // 3)] = rng.random() < 0.5
        # half the blobs get a jittered near-duplicate to exercise NMS
        if i % 2 == 0:
            d = np.zeros((H, W), dtype=bool)
            dr, dc = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            d[r + dr : min(H, r + dr + h), c + dc : min(W, c + dc + w)] = True
            out.append((d, "car", round(float(rng.random()), 2)))
        out.append((m, "car" if i % 3 else "building", round(float(rng.random()), 2)))
    # deliberate exact score tie on two disjoint masks
    t1 = np.zeros((H, W), dtype=bool)
    t1[1:5, 1:5] = True
    t2 = np.zeros((H, W), dtype=bool)
    t2[H - 5 : H - 1, W - 5 : W - 1] = True
    out.append((t1, "car", 0.5))
    out.append((t2, "car", 0.5))
    return out


def test_nms_matches_full_frame_reference():
    rng = np.random.default_rng(20260802)
    H = W = 160
    for trial in range(5):
        full = _random_blobs(rng, 25, H, W)
        for thr in (0.3, 0.5, 0.9):
            ref = _reference_nms_full_masks(full, thr)
            got = SamDetectClient._nms_masks(_items_from_full(full), thr)
            assert len(got) == len(ref)
            for (gcrop, gbb, glab, gsc), (rm, rlab, rsc) in zip(got, ref):
                assert (glab, gsc) == (rlab, rsc)
                rebuilt = np.zeros((H, W), dtype=bool)
                rebuilt[gbb[0] : gbb[2] + 1, gbb[1] : gbb[3] + 1] = gcrop
                assert np.array_equal(rebuilt, rm)


def test_encode_from_crop_matches_full_frame_encode():
    rng = np.random.default_rng(7)
    H, W = 130, 190
    for _ in range(20):
        m = np.zeros((H, W), dtype=bool)
        h, w = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        r, c = int(rng.integers(0, H - h)), int(rng.integers(0, W - w))
        m[r : r + h, c : c + w] = rng.random((h, w)) < 0.6
        if not m.any():
            continue
        rows, cols = m.any(axis=1), m.any(axis=0)
        r0, c0 = int(rows.argmax()), int(cols.argmax())
        r1 = int(H - 1 - rows[::-1].argmax())
        c1 = int(W - 1 - cols[::-1].argmax())
        via_crop = encode_mask_record(
            m[r0 : r1 + 1, c0 : c1 + 1], bbox_rcrc=(r0, c0, r1, c1)
        )
        assert via_crop == encode_mask_record(m)


def test_empty_item_is_dropped_not_crashed():
    live = np.ones((4, 6), dtype=bool)
    items = [
        (None, (0, 0, 0, 0), 0, "car", 0.9),
        (live, (10, 10, 13, 15), 24, "car", 0.4),
    ]
    kept = SamDetectClient._nms_masks(items, 0.5)
    assert [(bb, lab, sc) for (_c, bb, lab, sc) in kept] == [
        ((10, 10, 13, 15), "car", 0.4)
    ]


# ── End-to-end golden: _sahi_segment repack equals the reference repack ──


def _tile_det(mask_1008, label, score, oid):
    rec = encode_mask_record(mask_1008)
    assert rec is not None
    return {**rec, "label": label, "score": score, "object_id": oid}


def _tile_b64(rgb, bound):
    """Replay the client's own tile encode: the mock's content key.

    Calls the production encoder instead of re-implementing it. Re-implementing
    it keyed this fixture on PIL's default zlib level, so changing that level in
    the client failed both repack tests with a KeyError on a base64 blob rather
    than saying what actually moved.
    """
    r0, r1, c0, c1 = bound
    canvas, _scale, _pt, _pl = _letterbox_to_sam(rgb[r0:r1, c0:c1])
    return base64.b64encode(_png_wire_bytes(canvas)).decode("ascii")


def _build_repack_case(*, sahi_tile_workers: int = 1):
    """One SAHI frame (2 tiles), its wire payloads, and the reference
    repack. The rgb is a deterministic gradient so each tile's PNG is
    distinct — the content-keyed mock must never be degenerate."""
    H, W = 1008, 1512  # 1 row × 2 cols of native 1008 tiles: c0 ∈ {0, 504}
    min_area = 16
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["car"])
    client = SamDetectClient(
        concept_provider=provider,
        score_threshold=0.1,
        min_area_px=min_area,
        sahi=True,
        sahi_overlap=0.2,
        sahi_nms_iou=0.5,
        sahi_tile_workers=sahi_tile_workers,
    )
    bounds = client._tile_bounds(H, W, client.sahi_overlap)
    assert bounds == [(0, 1008, 0, 1008), (0, 1008, 504, 1512)]

    def tile_mask(r0, c0, r1, c1):
        m = np.zeros((1008, 1008), dtype=bool)
        m[r0:r1, c0:c1] = True
        return m

    # tile 0: one clean object + the seam object (frame cols 600:700)
    t0 = [
        _tile_det(tile_mask(100, 50, 140, 120), "car", 0.9, 0),
        _tile_det(tile_mask(300, 600, 340, 700), "car", 0.8, 1),
    ]
    # tile 1 (c0=504): seam duplicate (frame cols 600:700 → local 96:196,
    # lower score → NMS drops it), a tie pair, and a sub-min-area speck
    t1 = [
        _tile_det(tile_mask(300, 96, 340, 196), "car", 0.7, 0),
        _tile_det(tile_mask(500, 700, 560, 800), "car", 0.85, 1),
        _tile_det(tile_mask(700, 100, 760, 200), "building", 0.85, 2),
        _tile_det(tile_mask(10, 10, 12, 12), "car", 0.99, 3),  # 4 px < 16
    ]
    payloads = [
        {"detections": t0, "image_shape_hw": [1008, 1008]},
        {"detections": t1, "image_shape_hw": [1008, 1008]},
    ]

    # reference repack: full-frame remap → reference NMS → full-frame encode
    ref_full = []
    for (r0, r1, c0, c1), payload in zip(bounds, payloads):
        for det in payload["detections"]:
            wire = decode_mask_full(det, (1008, 1008))
            if int(wire.sum()) < min_area:
                continue
            fm = np.zeros((H, W), dtype=bool)
            fm[r0:r1, c0:c1] = wire
            ref_full.append((fm, str(det["label"]), float(det["score"])))
    ref_dets = []
    for i, (m, lab, sc) in enumerate(
        _reference_nms_full_masks(ref_full, client.sahi_nms_iou)
    ):
        rec = encode_mask_record(m)
        ref_dets.append({**rec, "object_id": i, "label": lab, "score": sc})
    assert len(ref_dets) == 4  # seam dedup dropped 1, speck dropped 1

    rr, cc = np.meshgrid(
        np.arange(H) % 256, np.arange(W) % 256, indexing="ij"
    )
    rgb = np.stack([rr, cc, (rr + cc) % 256], axis=-1).astype(np.uint8)
    payload_by_b64 = {
        _tile_b64(rgb, b): p for b, p in zip(bounds, payloads)
    }
    assert len(payload_by_b64) == 2, "tile PNGs must be distinct for keying"

    obs = Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=100.0,
        yaw_deg=0.0,
        world_window=(0.0, 0.0, 100.0, 100.0),
        rgb=rgb,
        ortho_view_corners_xy=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)),
    )
    return client, obs, payload_by_b64, ref_dets, (H, W)


def test_sahi_segment_repack_matches_reference(monkeypatch):
    client, obs, payload_by_b64, ref_dets, (H, W) = _build_repack_case()

    requested: list[str] = []

    def serve(*, picker, json_body):
        key = json_body["image_b64"]
        requested.append(key)
        return payload_by_b64[key]

    monkeypatch.setattr(
        "maps.perception.sam.detect_client._post_segment_avoid_stale_pool",
        serve,
    )
    captured = []
    monkeypatch.setattr(
        SamDetectClient,
        "_decode_payload_to_detections",
        lambda self, payload, **kw: captured.append(payload) or [],
    )
    client._sahi_segment(obs, ["car"])

    # vacuity floor: exactly the two expected tiles were requested
    assert sorted(requested) == sorted(payload_by_b64)
    assert captured, "SAHI path never reached the decode stage"
    assert captured[0] == {"detections": ref_dets, "image_shape_hw": [H, W]}


def test_sahi_tile_order_survives_out_of_order_completion(monkeypatch):
    """Force tile 1 to COMPLETE before tile 0 (workers=2, tile 0 gated on
    tile 1's completion) — the repack must still equal the bounds-order
    reference: collection order is bounds order, not completion order."""
    client, obs, payload_by_b64, ref_dets, (H, W) = _build_repack_case(
        sahi_tile_workers=2
    )
    idx_by_b64 = {k: i for i, k in enumerate(payload_by_b64)}

    completion_order: list[int] = []
    tile1_done = threading.Event()

    def serve(*, picker, json_body):
        key = json_body["image_b64"]
        idx = idx_by_b64[key]
        if idx == 0:
            assert tile1_done.wait(timeout=5.0), "tile 1 never completed"
        completion_order.append(idx)
        if idx == 1:
            tile1_done.set()
        return payload_by_b64[key]

    monkeypatch.setattr(
        "maps.perception.sam.detect_client._post_segment_avoid_stale_pool",
        serve,
    )
    captured = []
    monkeypatch.setattr(
        SamDetectClient,
        "_decode_payload_to_detections",
        lambda self, payload, **kw: captured.append(payload) or [],
    )
    client._sahi_segment(obs, ["car"])

    # vacuity floor: completion order really was inverted
    assert completion_order == [1, 0]
    assert captured[0] == {"detections": ref_dets, "image_shape_hw": [H, W]}


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
