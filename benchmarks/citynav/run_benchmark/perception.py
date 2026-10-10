"""Perception factory: the SAM segmenter."""

from __future__ import annotations

from benchmarks.citynav.run_benchmark.config import RunConfig
from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider
from maps.perception.sam.detect_client import SamDetectClient, set_color_palette


def make_perception(cfg: RunConfig) -> SamDetectClient:
    """The SAM client. Its class set is the canonicalized concept union of
    every active reasoner thread, pushed into the provider via
    ``set_concepts``; before the first push the provider is empty and
    ``SamDetectClient.detect`` returns no detections."""
    adapter = SamDetectClient(concept_provider=ReasonerUnionConceptProvider(), **cfg.sam)
    # Activate this camera's color calibration module-globally so BOTH the
    # detection path and the prior-percept enrichment path (which calls
    # _set_color_attr directly) share it.
    set_color_palette(adapter.color_palette)
    return adapter
