"""``_decode_payload_to_detections`` output, pinned value by value, serial and on a thread pool.

The case splits multi-component masks, drops a detection over NaN depth (its components and merged
fallback all fail projection), and hits ``max_per_step`` mid-detection. A change meant to move a value
here is a deliberate edit of the snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
from maps.perception.tests.regions import square_view
from maps.util.mask_codec import encode_mask_record

H = W = 128
MAX_PER_STEP = 9  # 8 two-component dets = 16 packs -> cap hits mid-det


def _det(mask, oid, score):
    rec = encode_mask_record(mask)
    assert rec is not None
    return {**rec, "label": "car", "score": score, "object_id": oid}


def _build_case():
    """8 disjoint two-component dets + one det on NaN depth (projection
    fails for its components AND its merged fallback -> dropped)."""
    depth = np.ones((H, W), dtype=np.float32)
    depth[100:, 100:] = np.nan
    rr, cc = np.meshgrid(np.arange(H) % 256, np.arange(W) % 256, indexing="ij")
    rgb = np.stack([rr, cc, (rr + cc) % 256], axis=-1).astype(np.uint8)

    def two_blocks(r, c):
        m = np.zeros((H, W), dtype=bool)
        m[r : r + 6, c : c + 6] = True
        m[r : r + 6, c + 10 : c + 16] = True  # second component
        return m

    dets = []
    oid = 0
    for r in (0, 16, 32, 48, 64, 80):
        for c in (0, 40):
            if len(dets) == 2:
                # det index 2 sits entirely in the NaN-depth corner: every
                # component and the merged fallback fail projection.
                dets.append(_det(two_blocks(100, 100), oid, 0.9))
                oid += 1
                if len(dets) == 9:
                    break
            dets.append(_det(two_blocks(r, c), oid, 0.9))
            oid += 1
            if len(dets) == 9:
                break
        if len(dets) == 9:
            break

    return {"detections": dets, "image_shape_hw": [H, W]}, square_view(rgb, depth)


def _run(decode_workers: int):
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["car"])
    client = SamDetectClient(
        concept_provider=provider,
        min_area_px=16,
        max_per_step=MAX_PER_STEP,
        decode_workers=decode_workers,
    )
    payload, obs = _build_case()
    return client._decode_payload_to_detections(
        payload,
        obs=obs,
        scale=1.0,
        pad_top=0,
        pad_left=0,
        orig_HW=(H, W),
    )


# (label, confidence, world_xy, mask.bbox, mask.area, track_id, pixel_area,
#  mask_pixel_count, color, color_rgb)
GOLDEN = [
    ("car", 0.9, (1.968504, 98.031496), (0, 0, 5, 5), 36, "sam3_1_img_0_c0", 36, 36, "black", "2,2,5"),
    ("car", 0.9, (9.84252, 98.031496), (0, 10, 5, 15), 36, "sam3_1_img_0_c1", 36, 36, "black", "2,12,15"),
    ("car", 0.9, (33.464567, 98.031496), (0, 40, 5, 45), 36, "sam3_1_img_1_c0", 36, 36, "black", "2,42,45"),
    ("car", 0.9, (41.338583, 98.031496), (0, 50, 5, 55), 36, "sam3_1_img_1_c1", 36, 36, "black", "2,52,55"),
    ("car", 0.9, (1.968504, 85.433071), (16, 0, 21, 5), 36, "sam3_1_img_3_c0", 36, 36, "black", "18,2,21"),
    ("car", 0.9, (9.84252, 85.433071), (16, 10, 21, 15), 36, "sam3_1_img_3_c1", 36, 36, "black", "18,12,31"),
    ("car", 0.9, (33.464567, 85.433071), (16, 40, 21, 45), 36, "sam3_1_img_4_c0", 36, 36, "black", "18,42,61"),
    ("car", 0.9, (41.338583, 85.433071), (16, 50, 21, 55), 36, "sam3_1_img_4_c1", 36, 36, "black", "18,52,71"),
    ("car", 0.9, (1.968504, 72.834646), (32, 0, 37, 5), 36, "sam3_1_img_5_c0", 36, 36, "black", "34,2,37"),
]


@pytest.mark.parametrize("workers", [1, 4])
def test_decode_output_matches_the_golden(workers):
    out = _run(workers)
    assert len(out) == len(GOLDEN)

    for i, (d, g) in enumerate(zip(out, GOLDEN)):
        label, conf, world_xy, bbox, mask_px, track_id, area, n_valid, color, rgb = g
        where = f"detection {i} ({track_id})"
        assert d.label == label, where
        assert round(float(d.confidence), 6) == conf, where
        # world_xy is the value a mask shift moves while every count holds.
        assert round(float(d.world_xy[0]), 6) == world_xy[0], where
        assert round(float(d.world_xy[1]), 6) == world_xy[1], where
        assert tuple(d.mask.bbox) == bbox, where
        assert d.mask.area == mask_px, where
        assert d.metadata["track_id"] == track_id, where
        assert int(d.metadata["pixel_area"]) == area, where
        assert int(d.metadata["mask_pixel_count"]) == n_valid, where
        assert d.attributes.get("color") == color, where
        assert d.attributes.get("color_rgb") == rgb, where
