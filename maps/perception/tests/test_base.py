"""The DetectedObject contract: the producer names its tier; no coarse class bucket rides along."""

from __future__ import annotations

import pytest

from maps.perception.detection import DetectedObject


def test_a_detection_names_its_tier_and_carries_no_bucket():
    with pytest.raises(TypeError):
        DetectedObject(world_xy=(1.0, 2.0), label="car", confidence=0.7)
    det = DetectedObject(
        world_xy=(1.0, 2.0), label="car", confidence=0.7, source_tier="sam3_1_image"
    )
    assert not hasattr(det, "class_hint")
    assert det.metadata == {} and det.attributes == {} and det.mask is None
