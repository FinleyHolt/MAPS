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
    # A region, not a frame-sized array with a box beside it: with separate
    # fields every consumer re-derives the box by scanning the whole frame,
    # which costs the SENSOR's resolution per detection. A region cannot be
    # separated from its box.
    mask: "MaskRegion | None" = None


class PerceptionAdapter(Protocol):
    def detect(
        self,
        obs: Observation,
        pose_xy: tuple[float, float],
        instruction_hint: str | None = None,
    ) -> list[DetectedObject]: ...
