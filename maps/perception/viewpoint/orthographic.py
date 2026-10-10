"""Orthographic-nadir picker for the ortho observation backend.

Upstream CityNav's view square is altitude-driven (side = 2*(z -
ground_level)) and yaw-rotated; pitch is always nadir. The picker's
job collapses to "fly directly over the target's
centroid at the in-band altitude the runtime already sampled".

Upstream CityNav itself has no view-pose optimization -- the agent moves
toward predicted goal xy and looks nadir. ``view_intent``,
``view_face``, ``view_from_fid``, and ``sensor`` are ignored on this
picker; the orthographic crop has no facade and no oblique angle to
optimize.

Pure stdlib + Python primitives: no numpy hot path, no torch, no GPU.
The dispatcher routes this picker in-process; the ortho crop has no GPU
work, so there is no external pose server to amortise a CUDA context.
"""

from __future__ import annotations

import logging
from typing import Sequence

from maps.perception.viewpoint.base import (
    ViewpointRequest,
    ViewpointResult,
)

log = logging.getLogger(__name__)


def pick_viewpoint(req: ViewpointRequest) -> ViewpointResult | None:
    """Fly directly over the target's centroid at the in-band altitude.

    Returns ``None`` when no feasible pose exists; the caller drops the
    observation. There is one picker because there is one observation
    backend -- the 2D nadir ortho crop. A 3D / non-nadir picker gets
    rebuilt under ``quadcopter.sim`` when a non-nadir sensor is needed.
    """
    x, y = req.target_xy
    altitude_m = float(req.altitude_hint_m)
    log.info(
        "nadir pick: target=%s pose=(%.1f, %.1f, %.1f) yaw=0.0",
        req.target_fid,
        float(x),
        float(y),
        altitude_m,
    )
    return ViewpointResult(
        pose_xy=(float(x), float(y)),
        altitude_m=altitude_m,
        yaw_deg=0.0,
        diag={
            "source": "orthographic_centroid",
            "target_xy": (float(x), float(y)),
            "altitude_m": altitude_m,
        },
    )
