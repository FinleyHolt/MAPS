"""DetectedObject defaults."""

from __future__ import annotations

from maps.perception.base import DetectedObject


def test_detected_object_defaults():
    det = DetectedObject(world_xy=(1.0, 2.0), label="car", confidence=0.7)
    assert det.class_hint == "object"
    assert det.source_tier == "synthetic"
    assert det.metadata == {}
    assert det.attributes == {}
