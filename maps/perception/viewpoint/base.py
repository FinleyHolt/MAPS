"""ViewpointRequest / ViewpointResult / TileBounds.

The shared types the nadir viewpoint picker reads and writes.

Robot-agnostic: this module lives in ``maps/perception/`` and does
not import from ``maps.search`` or ``benchmark`` per the
robot-agnostic rule in ``maps/perception/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from maps.perception.geometry.sensor_spec import SensorSpec


@dataclass(frozen=True)
class ViewpointRequest:
    """One observation's worth of input to ``pick_viewpoint``.

    Built by the caller (``tour_synth._pick_observation_pose``) from
    one ``Observation`` plus the resolver context. The dispatcher
    routes by ``backend``.

    Fields:
        target_fid: target feature id. CityRefer priors use
            ``cityref_<id>``; OSM/runtime priors may use ids such as
            ``road_22`` or ``detected_3``.
        target_xy: world (x, y) centroid of the target, which the
            orthographic picker flies straight over.
        altitude_hint_m: the per-episode in-band altitude the runtime
            will be flying at. The orthographic picker outputs this
            verbatim.
    """

    target_fid: str
    target_xy: tuple[float, float]
    altitude_hint_m: float


@dataclass(frozen=True)
class ViewpointResult:
    """One picker's verdict for one ``ViewpointRequest``.

    ``diag`` is opaque per-backend. The orthographic picker stamps a
    small payload with ``source = "orthographic_centroid"`` and the
    altitude / xy it picked, so the SPA / ``tour.json`` / audit
    contract holds.

    Camera aiming is intentionally NOT a field on this dataclass. The
    observation is an orthographic nadir crop, so an aim target is a place
    and ``tour_synth._attach_aim_targets`` resolves it after pose selection.
    Keeping the result a single non-union type keeps the schema flat.
    """

    pose_xy: tuple[float, float]
    altitude_m: float
    yaw_deg: float
    diag: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TileBounds:
    """Minimal description of a prebake tile's world-grid layout.

    Mirrors the shape of ``benchmarks.citynav.BlockAssets`` (``x_min``,
    ``y_min``, ``cell_size_m``, ``grid_shape``) without importing it, so this
    module stays robot-agnostic per ``maps/perception/README.md``. Lives here
    (shared viewpoint types) so the nadir orthographic picker and overlay can
    reference it directly.
    """

    x_min: float
    y_min: float
    cell_size_m: float
    grid_shape: tuple[int, int]
