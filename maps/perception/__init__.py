"""Robot-agnostic perception: observation crop + detection contract."""

from maps.perception.base import DetectedObject, GTObject, PerceptionAdapter
from maps.perception.observation import (
    BlockObservation,
    Observation,
    crop_observation,
    load_block_rasters,
)
from maps.perception.sam.detect_client import SamDetectClient

__all__ = [
    "BlockObservation",
    "DetectedObject",
    "GTObject",
    "Observation",
    "PerceptionAdapter",
    "SamDetectClient",
    "crop_observation",
    "load_block_rasters",
]

# The SAM engine lives with its serve (serving/sam_engine.py, its own env);
# the runtime adapter (sam.detect_client) talks to that server over HTTP.
