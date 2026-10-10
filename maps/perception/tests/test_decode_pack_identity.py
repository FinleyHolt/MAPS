"""Parallel decode/pack is byte-identical to serial across worker counts.

``_decode_payload_to_detections`` runs its per-detection decode (PASS 1)
and split/pack (PASS 2) on a thread pool and replays the append/cap/
fallback semantics serially in det order. A 1-worker pool IS the serial
semantics, so identity across ``decode_workers`` values pins the ordered
replay: same detections, same order, same cap truncation position.
"""

from __future__ import annotations

import numpy as np
import pytest

from maps.perception.observation import Observation
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
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

    payload = {"detections": dets, "image_shape_hw": [H, W]}
    obs = Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=100.0,
        yaw_deg=0.0,
        semantic=np.zeros((8, 8), dtype=int),
        height=None,
        fov_m=100.0,
        world_window=(0.0, 0.0, 100.0, 100.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=((0.0, 100.0), (100.0, 100.0), (100.0, 0.0), (0.0, 0.0)),
    )
    return payload, obs


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
        source_tier="sam3_1_image",
        frame_index_override=-1,
    )


@pytest.mark.parametrize("workers", [2, 4])
def test_decode_pack_parallel_matches_serial(workers):
    serial = _run(1)
    parallel = _run(workers)

    # vacuity floors: the cap really truncated mid-run, the case really
    # exercised multi-component splits, and the NaN-depth det really died.
    assert len(serial) == MAX_PER_STEP
    track_ids = [d.metadata["track_id"] for d in serial]
    assert any(t.endswith("_c1") for t in track_ids)
    assert not any(t.startswith("sam3_1_img_2") for t in track_ids)

    assert len(parallel) == len(serial)
    for a, b in zip(parallel, serial):
        assert a.label == b.label
        assert a.confidence == b.confidence
        assert a.metadata["track_id"] == b.metadata["track_id"]
        assert a.metadata["pixel_bbox"] == b.metadata["pixel_bbox"]
        assert a.world_xy == b.world_xy
        assert a.mask.bbox == b.mask.bbox
        assert np.array_equal(a.mask.crop, b.mask.crop)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
