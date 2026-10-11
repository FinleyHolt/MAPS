"""Per-detection decode cost scales with the objects, never with the frame.

The same detections decoded on a frame and on one with 16x the area: healthy reads a ratio near 1.0,
one reverted narrowed scan reads 4.3-4.5, a whole-frame scan per detection 15.8. If this goes flaky,
widen MAX_RATIO rather than delete the guard.
"""

from __future__ import annotations

import time

import numpy as np

from maps.perception.tests.regions import square_view
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam import detect_client as dc
from maps.perception.sam.detect_client import SamDetectClient
from maps.util.mask_codec import encode_mask_record

SMALL = 256
LARGE = 1024  # 16x the area
N_DETS = 24
DET_PX = 20
MAX_RATIO = 2.5


def _case(frame_px: int):
    depth = np.ones((frame_px, frame_px), dtype=np.float32)
    rgb = np.zeros((frame_px, frame_px, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    dets = []
    for i in range(N_DETS):
        m = np.zeros((frame_px, frame_px), dtype=bool)
        r, c = 10 + (i % 6) * 30, 10 + (i // 6) * 30
        m[r : r + DET_PX, c : c + DET_PX] = True
        rec = encode_mask_record(m)
        assert rec is not None
        dets.append({**rec, "label": "car", "score": 0.9, "object_id": i})
    return {"detections": dets, "image_shape_hw": [frame_px, frame_px]}, square_view(rgb, depth)


def _per_detection_seconds(frame_px: int, reps: int = 3) -> tuple[float, int]:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["car"])
    client = SamDetectClient(concept_provider=provider)
    payload, obs = _case(frame_px)
    best = float("inf")
    out: list = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = client._decode_payload_to_detections(
            payload,
            obs=obs,
            scale=1.0,
            pad_top=0,
            pad_left=0,
            orig_HW=(frame_px, frame_px),
        )
        best = min(best, time.perf_counter() - t0)
    return best / max(len(out), 1), len(out)


def test_per_detection_decode_cost_does_not_track_frame_area(monkeypatch):
    monkeypatch.setattr(dc, "_DECODE_WORKERS", 1)  # serial: this measures work, not scheduling
    monkeypatch.setattr(dc, "MIN_AREA_PX", 16)
    monkeypatch.setattr(dc, "MAX_PER_STEP", 999)
    small, n_small = _per_detection_seconds(SMALL)
    large, n_large = _per_detection_seconds(LARGE)

    assert n_small == n_large == N_DETS, (n_small, n_large)

    ratio = large / small
    assert ratio < MAX_RATIO, (
        f"per-detection decode cost grew {ratio:.1f}x for {(LARGE / SMALL) ** 2:.0f}x "
        f"the frame area ({small * 1e6:.0f} -> {large * 1e6:.0f} us/detection). "
        "Something in the per-detection path is scanning the whole frame again "
        "— look for a frame-sized array or a np.where/any/sum over one."
    )
