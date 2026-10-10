"""What ``_decode_payload_to_detections`` produces, pinned value by value.

The per-detection stages of the decoder are being reworked to stop building a
frame-sized canvas for every mask. Every one of those changes is supposed to be
a pure representation change, and the failure mode when it is not is a mask
that shifts by a few pixels — which moves ``world_xy`` a little, keeps every
count identical, and shows up as a slightly worse SR weeks later.

Nothing else in the suite pins the decoder's OUTPUT: the neighbouring identity
test compares two runs of the same code against each other, so it stays green
through any change that is merely consistently wrong. This one carries the
values, captured from the build before the rework, and it is a snapshot on
purpose — if a change is meant to move a number here, that has to be a
deliberate edit with a reason, not a green test.

The fixture is shared with ``test_decode_pack_identity``: multi-component
splits, a detection over NaN depth whose components and merged fallback all
fail projection, and a ``max_per_step`` cap that truncates mid-detection.
"""

from __future__ import annotations

import numpy as np

from maps.perception.tests.test_decode_pack_identity import MAX_PER_STEP, _run

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


def test_decode_output_matches_the_golden():
    out = _run(1)
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


def test_the_golden_case_still_exercises_what_it_claims_to():
    """Vacuity floor for the snapshot above.

    A fixture that quietly stopped splitting, stopped dropping the
    NaN-depth detection, or stopped hitting the cap would still match a
    golden captured from that same degraded fixture. These are the three
    behaviours the snapshot exists to protect."""
    out = _run(1)
    track_ids = [d.metadata["track_id"] for d in out]

    assert len(out) == MAX_PER_STEP, "cap no longer truncates"
    assert any(t.endswith("_c1") for t in track_ids), "no multi-component split"
    assert not any(
        t.startswith("sam3_1_img_2") for t in track_ids
    ), "the NaN-depth detection survived"
    # Masks are real regions: non-empty, and a small part of their frame.
    for d in out:
        assert 0 < d.mask.area <= d.mask.crop.size
        frame_px = d.mask.frame_shape[0] * d.mask.frame_shape[1]
        assert d.mask.crop.size < frame_px / 4
