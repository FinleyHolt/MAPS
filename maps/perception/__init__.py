"""Robot-agnostic perception: observation crop + detection contract."""

from maps.perception.base import DetectedObject
from maps.perception.observation import Observation, crop_observation
from maps.perception.sam.detect_client import SamDetectClient

__all__ = [
    "DetectedObject",
    "Observation",
    "SamDetectClient",
    "crop_observation",
]

# The SAM engine lives with its serve (serving/sam_engine.py, its own env);
# the runtime adapter (sam.detect_client) talks to that server over HTTP.
