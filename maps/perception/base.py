"""Perception contract: the detection dataclass.

The segmenter (:class:`maps.perception.sam.detect_client.SamDetectClient`)
converts an observation + pose into a list of detected objects, one crop per
call. Each detection carries enough metadata for a caller to promote it into
a spatial graph, wrap it as evidence, or dedup it across steps — none of
which is perception's job — and the world tracker stamps the stable
``metadata["track_id"]`` the graph upsert uses to re-identify objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.util.mask_region import MaskRegion


@dataclass
class DetectedObject:
    """A single runtime-detected object in world coordinates."""

    world_xy: tuple[float, float]  # Centroid in world frame (meters)
    label: str  # Free-form class-less label ("car", "building", ...)
    confidence: float  # [0, 1]
    source_tier: str  # the producer: "sam3_1_image"
    metadata: dict = field(default_factory=dict)  # track_id, sam3_1_object_id, tile_cells, ...
    attributes: dict[str, str] = field(default_factory=dict)
    # Semantic attributes per detection (the colour read). The graph upsert
    # folds them into a node's semantic_attributes as {key: {value: 1}} counters.

    # The segmentation mask, as a crop plus where it sits in the frame
    # (see ``maps.util.mask_region``).
    #
    # A region, not a frame-sized array with a box beside it: with separate
    # fields every consumer re-derives the box by scanning the whole frame,
    # which costs the SENSOR's resolution per detection. A region cannot be
    # separated from its box.
    mask: "MaskRegion | None" = None
