"""DetectedObject defaults and round-trip."""

from __future__ import annotations

from dataclasses import asdict

from maps.perception.base import DetectedObject


def test_detected_object_defaults():
    det = DetectedObject(world_xy=(1.0, 2.0), label="car", confidence=0.7)
    assert det.class_hint == "object"
    assert det.source_tier == "synthetic"
    assert det.metadata == {}
    assert det.attributes == {}


def test_detected_object_attributes_round_trip():
    det = DetectedObject(
        world_xy=(0.0, 0.0),
        label="truck",
        confidence=0.9,
        attributes={"color": "red", "text": "ACME"},
    )
    d = asdict(det)
    assert d["attributes"] == {"color": "red", "text": "ACME"}
