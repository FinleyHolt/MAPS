"""Perception contract: detection dataclass, adapter protocol, GT seed type.

Perception layers convert an observation + pose (+ optional instruction hint)
into a list of detected objects. Each detection carries enough metadata for a
caller to promote it into a spatial graph, wrap it as evidence, or dedup it
across steps — none of which is perception's job.

All adapters are callable via ``adapter.detect(obs, pose_xy, instruction_hint)``.
The GT-perfect tier reads a ground-truth landmark list; the real tier
(:class:`perception.sam.detect_client.SamDetectClient`) segments one crop per
call and returns masks carrying the world tracker's stable
``metadata["track_id"]``, which the graph upsert uses to re-identify objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from maps.perception.observation import Observation
from maps.util.mask_region import MaskRegion


@dataclass
class DetectedObject:
    """A single runtime-detected object in world coordinates."""

    world_xy: tuple[float, float]  # Centroid in world frame (meters)
    label: str  # Free-form class-less label ("car", "building", ...)
    confidence: float  # [0, 1]
    class_hint: str = "object"  # Coarse bucket for dedup/re-id (node class = fine label)
    description: str | None = None  # Free-form descriptor (optional)
    embedding: list[float] | None = None  # Optional vector for dedup / re-id
    source_tier: str = "synthetic"  # "synthetic" | "sam" | ...
    metadata: dict = field(default_factory=dict)  # provenance, bbox, track_id, ...
    attributes: dict[str, str] = field(default_factory=dict)
    # Open-vocab semantic attributes per detection (color, material, text_span,
    # ...). Callers typically fold these into a graph node's
    # semantic_attributes as {key: {value: 1}} counters. Synthetic perception
    # leaves this empty; real adapters populate it.

    # Optional segmentation mask, as a crop plus where it sits in the frame
    # (see ``maps.util.mask_region``). Populated by segmentation-capable
    # adapters (real SAM/SegFormer paths or the synthetic disc-mask mode).
    # When present, the downstream geolocator projects it onto the tile grid
    # and the runtime upsert consumes the resulting ground mask.
    #
    # A region, not a frame-sized array with a box beside it: the two used to
    # be separate fields and every consumer re-derived the box by scanning the
    # whole frame, which cost the SENSOR's resolution per detection. A region
    # cannot be separated from its box, so that cannot come back.
    mask: "MaskRegion | None" = None


@dataclass
class GTObject:
    """A single ground-truth landmark used to seed the synthetic detector.

    Robot-agnostic: any benchmark that has a GT landmark set (CityNav,
    Matterport, a warehouse layout, etc.) adapts its native schema into a
    list[GTObject] at its own seam. Perception never sees the native schema.
    """

    world_xy: tuple[float, float]  # Centroid in world frame (meters)
    label: str  # Display / readable name
    class_hint: str  # Coarse class bucket that downstream graphs use
    object_type: str  # Raw source-dataset type string (for metadata)
    id: int | str  # Stable source-dataset identifier
    # GT-PERFECT-only extensions (both default-valued, so every other
    # GTObject construction is unaffected). ``contour`` is the object's 2D world-space footprint
    # polygon (CityRefer ``contour``), rasterized into the true mask by the
    # perfect sensor; ``attributes`` carries the offline GT color table row
    # ({"color", "color_conf"}) injected onto the detection so it flows to the
    # reasoner's color surface exactly like a real perceived attribute.
    contour: list[tuple[float, float]] | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    # Annotated object extent (dx, dy, dz) in meters (CityRefer ``dimension``).
    # Consumed ONLY by the perfect sensor's WORLD_CONTOUR tightness gate (a
    # contour whose area far exceeds dx*dy is a loose annotation hull, not a
    # footprint); never surfaced to the reasoner. None when the source has no
    # per-object extent.
    dimension: tuple[float, float, float] | None = None


class PerceptionAdapter(Protocol):
    def detect(
        self,
        obs: Observation,
        pose_xy: tuple[float, float],
        instruction_hint: str | None = None,
    ) -> list[DetectedObject]: ...
