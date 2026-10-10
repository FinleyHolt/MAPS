"""Decode cost must scale with the OBJECTS, never with the frame.

This is the guard, not another equivalence test. Every whole-frame bug in this
path had the same signature — a per-detection stage touching every pixel of the
sensor's frame to read an object covering a few hundred of them — and every one
of them was invisible to the suite, because a whole-frame scan is not wrong. It
returns the right answer. It just costs the frame.

So the property under test is the cost curve. The same detections, at the same
pixel positions, decoded on a small frame and on one with 16x the area: if
per-detection cost tracks the frame, something is scanning it again.

Measured on the same box, 256 px to 2048 px — 64x the pixels:

    before the rework    540 -> 8556 us/detection   (x15.8)
    interior crop-native 429 ->  666 us/detection   (x1.55)
    mask field a region  421 ->  428 us/detection   (x1.02)

The middle row still carried one deliberate canvas per detection, because
``DetectedObject.pixel_mask`` was a frame-sized contract. That field is a
``MaskRegion`` now, so no frame-sized array is built on the path at all and
the curve is flat.

A timing assertion earns its place here because the margin is wide and was
checked in BOTH directions: healthy reads ~1.0, deliberately reverting a
single narrowed scan reads 4.3-4.5, and reverting all of them read 15.8. The
threshold sits at 2.5, between the two. It is not a benchmark and must never be
tightened into one; if it ever goes flaky, widen the threshold rather than
delete the guard, because deleting it puts the whole class of bug back out of
sight.
"""

from __future__ import annotations

import time

import numpy as np

from maps.perception.observation import Observation
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient
from maps.util.mask_codec import encode_mask_record

SMALL = 256
LARGE = 1024  # 16x the area
N_DETS = 24
DET_PX = 20
# Measured band on this path, best-of-3 per arm: healthy reads ~1.0 (the curve
# is flat once nothing builds a frame-sized array), and reverting a SINGLE one
# of the narrowed scans reads 4.3-4.5. 2.5 sits between them with room on both
# sides. Reverting all of them read 15.8.
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
    obs = Observation(
        pose_xy=(0.0, 0.0),
        altitude_m=100.0,
        yaw_deg=0.0,
        world_window=(0.0, 0.0, 100.0, 100.0),
        rgb=rgb,
        depth=depth,
        ortho_view_corners_xy=(
            (0.0, 100.0),
            (100.0, 100.0),
            (100.0, 0.0),
            (0.0, 0.0),
        ),
    )
    return {"detections": dets, "image_shape_hw": [frame_px, frame_px]}, obs


def _per_detection_seconds(frame_px: int, reps: int = 3) -> tuple[float, int]:
    provider = ReasonerUnionConceptProvider()
    provider.set_concepts(["car"])
    client = SamDetectClient(
        concept_provider=provider,
        min_area_px=16,
        max_per_step=999,
        decode_workers=1,  # serial: this measures work, not scheduling
    )
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


def test_per_detection_decode_cost_does_not_track_frame_area():
    small, n_small = _per_detection_seconds(SMALL)
    large, n_large = _per_detection_seconds(LARGE)

    # Vacuity floor: both frames must have decoded the SAME detections, or the
    # ratio compares different amounts of work and means nothing.
    assert n_small == n_large == N_DETS, (n_small, n_large)

    ratio = large / small
    assert ratio < MAX_RATIO, (
        f"per-detection decode cost grew {ratio:.1f}x for {(LARGE / SMALL) ** 2:.0f}x "
        f"the frame area ({small * 1e6:.0f} -> {large * 1e6:.0f} us/detection). "
        "Something in the per-detection path is scanning the whole frame again "
        "— look for a frame-sized array or a np.where/any/sum over one."
    )


def test_the_detections_really_are_a_small_part_of_the_large_frame():
    """The guard only means something if an object is a tiny fraction of the
    frame — that ratio IS the bug's leverage."""
    payload, _ = _case(LARGE)
    covered = N_DETS * DET_PX * DET_PX
    assert covered < (LARGE * LARGE) / 100
    assert len(payload["detections"]) == N_DETS
