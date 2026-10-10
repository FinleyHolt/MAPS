"""Where to fly to see a target.

One observation backend means one picker: ``pick_viewpoint`` returns the
nadir pose over a target's centroid at the episode's in-band altitude, or
``None`` when no feasible pose exists. A 3D / non-nadir picker gets rebuilt
under ``quadcopter.sim`` when a non-nadir sensor is needed.

``overlay`` sits alongside it because the catalog-prior footprint it
projects is what the picker's crop is read against.
"""

from __future__ import annotations

from maps.perception.viewpoint.base import (
    TileBounds,
    ViewpointRequest,
    ViewpointResult,
)
from maps.perception.viewpoint.orthographic import pick_viewpoint

__all__ = [
    "TileBounds",
    "ViewpointRequest",
    "ViewpointResult",
    "pick_viewpoint",
]
