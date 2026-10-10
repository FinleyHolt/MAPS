"""View-frame geometry helpers shared by belief and maps.sws.

Single home for converting world-frame offsets into the drone's body frame so
view-relative reasoning ("left of X", "behind Y") has a consistent convention.

Convention
----------
- World frame: x is east-positive meters, y is north-positive meters.
- Yaw: degrees, 0 = facing north (+y), 90 = facing east (+x). Right-hand rule
  about the up axis, so increasing yaw rotates the heading clockwise when
  viewed from above.
- Body frame: forward is the drone's heading axis; left is forward rotated
  90 degrees counter-clockwise (so a north-facing drone sees +y as forward
  and -x as left).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViewFrame:
    """A node's position in the drone's body frame at one moment in time.

    Auto-populated for every prior-graph node each turn the compiler runs;
    invalidated and recomputed on every new ``RobotPose``. Cheap to compute
    (vectorised over all nodes per turn) so the prior graph itself stays
    frozen — these annotations live in a sidecar dict, never on the
    ``MapObject``.
    """

    forward_m: float  # body-frame forward distance, drone -> node centroid
    left_m: float  # body-frame left distance, signed
    range_m: float  # planar euclidean distance, always >= 0
    bearing_deg: float  # 0 = dead ahead, 90 = directly left, 180 = behind, 270 = right
    quadrant: str  # "front-left" | "front-right" | "back-left" | "back-right" | "at"






def compute_view_frames(
    centroids_rowcol,
    *,
    drone_x_m: float,
    drone_y_m: float,
    yaw_deg: float,
    tile_origin_xy: tuple[float, float],
    cell_size_m: float,
    grid_height: int,
) -> list[ViewFrame]:
    """Vectorised projection of N node centroids into the drone's body frame.

    ``centroids_rowcol`` is an iterable of ``(row, col)`` integer pairs. Returns
    a list of ``ViewFrame`` aligned with the input order. Grid row 0 is the
    south edge (= y_min) and col 0 is the west edge (= x_min); this matches
    the bake and the satellite ingest (verified end-to-end against
    CityRefer object positions). One matmul + a few comparisons per call;
    sub-millisecond for thousands of nodes.

    ``grid_height`` is accepted for backward compatibility with callers that
    still pass it; it is not used in the projection.
    """
    arr = np.asarray(list(centroids_rowcol), dtype=np.float32)
    if arr.size == 0:
        return []
    rows = arr[:, 0]
    cols = arr[:, 1]
    x0, y0 = tile_origin_xy
    # Cell-center world coords from tile origin. Forward affine in both
    # axes: row 0 is south (= y_min), col 0 is west (= x_min).
    cell_world_x = x0 + (cols + 0.5) * cell_size_m
    cell_world_y = y0 + (rows + 0.5) * cell_size_m
    _ = grid_height  # retained kwarg; intentionally unused
    dx = cell_world_x - drone_x_m
    dy = cell_world_y - drone_y_m
    yaw_rad = math.radians(yaw_deg)
    sin_y = math.sin(yaw_rad)
    cos_y = math.cos(yaw_rad)
    forward = dx * sin_y + dy * cos_y
    left = -dx * cos_y + dy * sin_y
    range_m = np.sqrt(dx * dx + dy * dy)
    # Body-frame bearing: 0 = ahead, 90 = left, 180 = behind, 270 = right.
    # atan2(left, forward) gives counter-clockwise from forward axis;
    # convert to degrees in [0, 360).
    bearing = np.degrees(np.arctan2(left, forward))
    bearing = np.where(bearing < 0, bearing + 360.0, bearing)
    out: list[ViewFrame] = []
    for f, l, r, b in zip(forward, left, range_m, bearing):
        if r < cell_size_m * 0.5:
            quad = "at"
        elif f >= 0 and l >= 0:
            quad = "front-left"
        elif f >= 0 and l < 0:
            quad = "front-right"
        elif f < 0 and l >= 0:
            quad = "back-left"
        else:
            quad = "back-right"
        out.append(
            ViewFrame(
                forward_m=float(f),
                left_m=float(l),
                range_m=float(r),
                bearing_deg=float(b),
                quadrant=quad,
            )
        )
    return out
