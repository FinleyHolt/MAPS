"""DetectedObject + GTObject defaults and round-trip."""

from __future__ import annotations

from dataclasses import asdict

from maps.perception.base import DetectedObject, GTObject


def test_detected_object_defaults():
    det = DetectedObject(world_xy=(1.0, 2.0), label="car", confidence=0.7)
    assert det.class_hint == "object"
    assert det.description is None
    assert det.embedding is None
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


def test_gt_object_fields():
    gt = GTObject(
        world_xy=(10.0, 20.0),
        label="Blue Car",
        class_hint="vehicles",
        object_type="Car",
        id=42,
    )
    assert gt.world_xy == (10.0, 20.0)
    assert gt.label == "Blue Car"
    assert gt.class_hint == "vehicles"
    assert gt.object_type == "Car"
    assert gt.id == 42


def test_gt_object_id_accepts_string():
    gt = GTObject(
        world_xy=(0.0, 0.0),
        label="x",
        class_hint="object",
        object_type="Generic",
        id="scene-12/obj-7",
    )
    assert gt.id == "scene-12/obj-7"
