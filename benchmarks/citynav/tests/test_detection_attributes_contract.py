"""Contract pin: DetectedObject.attributes survive the detection → graph hop.

Perception emits `DetectedObject.attributes` as `dict[str, str]` and the
belief engine's ``_detection_to_mapobject`` helper converts that into
the `MapObject.semantic_attributes` counter shape the runtime graph
accumulates. These tests pin that contract so a VLM drift (or a silent
rename) cannot quietly break downstream consumers.
"""

from __future__ import annotations


from maps.mission.episode_world import _detection_to_mapobject
from maps.perception.detection import DetectedObject


_BOUNDS = (0.0, 0.0)


def _make_detection(*, world_xy=(10.0, 20.0), attributes=None, label="truck") -> DetectedObject:
    return DetectedObject(
        world_xy=world_xy,
        label=label,
        confidence=0.8,
        source_tier="sam3_1_image",
        attributes=attributes or {},
    )


def test_open_vocab_attributes_map_to_counter_format():
    """{"color": "red"} must land as {"color": {"red": 1}} on the node."""
    det = _make_detection(attributes={"color": "red", "material": "metal"})
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0)

    assert node.semantic_attributes == {
        "color": {"red": 1},
        "material": {"metal": 1},
        "sam_label": {"truck": 1},
    }


def test_arbitrary_keys_preserved_without_whitelist():
    """The schema must not cherry-pick known keys; anything the VLM reads flows through."""
    det = _make_detection(
        attributes={
            "color": "blue",
            "material": "stone",
            "text": "ACME",
            "roof_shape": "gable",
            "floors": "3",
        }
    )
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0)
    assert set(node.semantic_attributes) == {
        "color",
        "material",
        "text",
        "roof_shape",
        "floors",
        "sam_label",
    }


def test_empty_attributes_still_records_sam_label():
    det = _make_detection(attributes={})
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0)
    assert node.semantic_attributes == {"sam_label": {"truck": 1}}


def test_runtime_node_carries_fine_label():
    """A runtime detection's graph node carries its FINE SAM label as
    ``class_name``."""
    det = _make_detection(label="white car")
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0)
    assert node.class_name == "white car"


def test_runtime_node_falls_back_to_object_when_unlabeled():
    """No SAM label -> the node is named "object"."""
    det = _make_detection(label="")
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0)
    assert node.class_name == "object"


def test_label_and_attributes_land_in_semantic_attributes():
    det = _make_detection(label="fire_hydrant", attributes={"color": "yellow"})
    node = _detection_to_mapobject(det, _BOUNDS, 5.0)
    assert node.semantic_attributes == {
        "color": {"yellow": 1},
        "sam_label": {"fire_hydrant": 1},
    }
