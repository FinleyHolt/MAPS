"""SAHI tiling and cross-tile NMS against a full-frame reference.

The client carries each tile detection as a region (a bbox crop placed in the
frame) instead of a frame-sized mask; these tests hold the kept set, its order
and its pixels equal to the full-frame pipeline, which lives here as the
reference implementation.
"""

from __future__ import annotations

import base64
import threading

import numpy as np
import pytest

from maps.perception.tests.regions import full, square_view
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam import detect_client as dc
from maps.perception.sam.detect_client import SamDetectClient, _png_wire_bytes
from maps.util.mask_codec import decode_mask_crop, encode_mask_record
from maps.util.mask_region import MaskRegion


def _reference_nms_full_masks(items, iou_thr):
    """Greedy IoU-NMS over FULL-FRAME masks: the reference semantics."""
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


def test_nms_matches_full_frame_reference(monkeypatch):
    rng = np.random.default_rng(20260802)
    H = W = 160
    for trial in range(5):
        masks = _random_blobs(rng, 25, H, W)
        items = [(MaskRegion.from_full(m), lab, sc) for m, lab, sc in masks]
        for thr in (0.3, 0.5, 0.9):
            ref = _reference_nms_full_masks(masks, thr)
            monkeypatch.setattr(dc, "SAHI_NMS_IOU", thr)
            got = SamDetectClient._nms_masks(items)
            assert [(lab, sc) for _r, lab, sc in got] == [(lab, sc) for _m, lab, sc in ref]
            for (region, _l, _s), (rm, _rl, _rs) in zip(got, ref):
                assert np.array_equal(full(region), rm)


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


# ── End-to-end: the masks _segment decodes equal the full-frame reference ──


def _tile_det(mask_1008, label, score, oid):
    rec = encode_mask_record(mask_1008)
    assert rec is not None
    return {**rec, "label": label, "score": score, "object_id": oid}


def _tile_b64(rgb, bound):
    """The client's own tile encode: the mock's content key."""
    r0, r1, c0, c1 = bound
    return base64.b64encode(_png_wire_bytes(np.ascontiguousarray(rgb[r0:r1, c0:c1]))).decode("ascii")


def _build_tile_case():
    """One frame of two tiles, their wire payloads, and the reference kept
    masks. The rgb is a deterministic gradient so each tile's PNG is
    distinct and the content-keyed mock is never degenerate."""
    H, W = 1008, 1512  # 1 row x 2 cols of native 1008 tiles: c0 in {0, 504}
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["car"])
    client = SamDetectClient(concept_provider=provider)
    bounds = client._tile_bounds(H, W)
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
    # tile 1 (c0=504): seam duplicate (frame cols 600:700 -> local 96:196,
    # lower score -> NMS drops it), a tie pair, and a sub-min-area speck
    t1 = [
        _tile_det(tile_mask(300, 96, 340, 196), "car", 0.7, 0),
        _tile_det(tile_mask(500, 700, 560, 800), "car", 0.85, 1),
        _tile_det(tile_mask(700, 100, 760, 200), "building", 0.85, 2),
        _tile_det(tile_mask(10, 10, 12, 12), "car", 0.99, 3),  # 4 px < MIN_AREA_PX
    ]
    payloads = [
        {"detections": t0, "image_shape_hw": [1008, 1008]},
        {"detections": t1, "image_shape_hw": [1008, 1008]},
    ]

    ref_full = []
    for (r0, r1, c0, c1), payload in zip(bounds, payloads):
        for det in payload["detections"]:
            fm = np.zeros((H, W), dtype=bool)
            pr0, pc0, pr1, pc1 = det["pixel_bbox_rcrc"]
            fm[r0 + pr0 : r0 + pr1 + 1, c0 + pc0 : c0 + pc1 + 1] = decode_mask_crop(det)
            if int(fm.sum()) >= dc.MIN_AREA_PX:
                ref_full.append((fm, str(det["label"]), float(det["score"])))
    ref = _reference_nms_full_masks(ref_full, dc.SAHI_NMS_IOU)
    assert len(ref) == 4  # seam dedup dropped 1, speck dropped 1

    rr, cc = np.meshgrid(np.arange(H) % 256, np.arange(W) % 256, indexing="ij")
    rgb = np.stack([rr, cc, (rr + cc) % 256], axis=-1).astype(np.uint8)
    payload_by_b64 = {_tile_b64(rgb, b): p for b, p in zip(bounds, payloads)}
    assert len(payload_by_b64) == 2, "tile PNGs must be distinct for keying"
    return client, square_view(rgb), payload_by_b64, ref


def _decoded(monkeypatch, client, obs):
    captured = []
    monkeypatch.setattr(SamDetectClient, "_detections", lambda self, obs, masks: captured.append(masks) or [])
    client._segment(obs, ["car"])
    assert captured, "the tiles never reached the decode"
    return captured[0]


def _assert_matches_reference(masks, ref):
    assert [(oid, lab, sc) for _r, lab, oid, sc in masks] == [(i, lab, sc) for i, (_m, lab, sc) in enumerate(ref)]
    for (region, _l, _o, _s), (rm, _rl, _rs) in zip(masks, ref):
        assert np.array_equal(full(region), rm)


def test_segment_keeps_what_the_full_frame_reference_keeps(monkeypatch):
    client, obs, payload_by_b64, ref = _build_tile_case()
    requested: list[str] = []

    def serve(json_body):
        requested.append(json_body["image_b64"])
        return payload_by_b64[json_body["image_b64"]]

    monkeypatch.setattr(dc, "_post_segment", serve)
    masks = _decoded(monkeypatch, client, obs)
    assert sorted(requested) == sorted(payload_by_b64)
    _assert_matches_reference(masks, ref)


def test_tile_order_survives_out_of_order_completion(monkeypatch):
    """Tile 1 COMPLETES before tile 0 (tile 0 gated on it); the kept masks
    still follow bounds order, not completion order."""
    client, obs, payload_by_b64, ref = _build_tile_case()
    idx_by_b64 = {k: i for i, k in enumerate(payload_by_b64)}
    completion_order: list[int] = []
    tile1_done = threading.Event()

    def serve(json_body):
        idx = idx_by_b64[json_body["image_b64"]]
        if idx == 0:
            assert tile1_done.wait(timeout=5.0), "tile 1 never completed"
        completion_order.append(idx)
        if idx == 1:
            tile1_done.set()
        return payload_by_b64[json_body["image_b64"]]

    monkeypatch.setattr(dc, "_post_segment", serve)
    masks = _decoded(monkeypatch, client, obs)
    assert completion_order == [1, 0]
    _assert_matches_reference(masks, ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
