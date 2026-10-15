"""Perception contract: the detection dataclass.

The segmenter (:class:`maps.perception.sam.detect_client.SamDetectClient`)
turns an observation into detected objects; the world tracker stamps the
stable ``metadata["track_id"]`` the graph upsert uses to re-identify them.
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
    metadata: dict = field(default_factory=dict)  # track_id, sam3_1_object_id, world_contour, ...
    attributes: dict[str, str] = field(default_factory=dict)
    # Semantic attributes per detection (the colour read). The graph upsert
    # folds them into a node's semantic_attributes as {key: {value: 1}} counters.

    # The segmentation mask, as a crop plus where it sits in the frame
    # (``maps.util.mask_region``), so no consumer scans the whole frame.
    mask: "MaskRegion | None" = None
