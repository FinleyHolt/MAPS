"""Contract pin: DetectedObject.attributes survive the detection → graph hop.

Perception emits `DetectedObject.attributes` as `dict[str, str]` and the
belief engine's ``_detection_to_mapobject`` helper converts that into
the `MapObject.semantic_attributes` counter shape the runtime graph
accumulates. These tests pin that contract so a VLM drift (or a silent
rename) cannot quietly break downstream consumers.
"""

from __future__ import annotations


from maps.search.search_loop import (
    _class_to_roles_from_specs,
    _detection_to_mapobject,
    _lookup_class_roles,
)
from maps.perception.base import DetectedObject


_BOUNDS = (0.0, 0.0)


def _make_detection(
    *, world_xy=(10.0, 20.0), attributes=None, label="truck"
) -> DetectedObject:
    return DetectedObject(
        world_xy=world_xy,
        label=label,
        confidence=0.8,
        class_hint="object",
        attributes=attributes or {},
    )


def test_open_vocab_attributes_map_to_counter_format():
    """{"color": "red"} must land as {"color": {"red": 1}} on the node."""
    det = _make_detection(attributes={"color": "red", "material": "metal"})
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0, now=100.0)

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
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0, now=0.0)
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
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0, now=0.0)
    assert node.semantic_attributes == {"sam_label": {"truck": 1}}


def test_runtime_node_carries_fine_label_and_resolves_target_anchor_role():
    """A runtime detection's graph node must carry its FINE SAM label as
    ``class_name`` (not the coarse ``class_hint`` bucket) so the per-task
    target/anchor split resolves its role. This is the lever that lets the
    in-flight commit prefer a target-class detection over a pure anchor.
    """
    # The node class is the fine label, not the coarse "object" bucket.
    det = _make_detection(label="white car")
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0, now=0.0)
    assert node.class_name == "white car"

    # The reasoner's Stage-1 split (its own sam_concepts roles).
    split = _class_to_roles_from_specs(
        [
            {"class": "car", "roles": ["target"]},
            {"class": "road", "roles": ["anchor"]},
        ]
    )
    # The fine label resolves to the target role via substring match on the
    # head noun ("car" in "white car"); a pure-anchor class stays anchor; an
    # unmatched class returns no role.
    assert "target" in _lookup_class_roles(node.class_name, split)
    assert _lookup_class_roles("road", split) == ("anchor",)
    assert _lookup_class_roles("river", split) == ()


def test_runtime_node_falls_back_to_class_hint_when_unlabeled():
    """No SAM label -> the coarse class_hint bucket still names the node."""
    det = _make_detection(label="")
    node = _detection_to_mapobject(det, _BOUNDS, cell_size_m=5.0, now=0.0)
    assert node.class_name == "object"


def test_label_and_source_tier_survive_onto_tags():
    det = _make_detection(label="fire_hydrant", attributes={"color": "yellow"})
    det.source_tier = "vlm"
    node = _detection_to_mapobject(det, _BOUNDS, 5.0, now=0.0)
    assert node.tags["label"] == "fire_hydrant"
    assert node.tags["source_tier"] == "vlm"
    assert node.tags == {"label": "fire_hydrant", "source_tier": "vlm"}
    assert node.semantic_attributes == {
        "color": {"yellow": 1},
        "sam_label": {"fire_hydrant": 1},
    }
