"""RobotPose: the drone's world-frame state at compile time.

Carried on ``CompilerInput.robot_pose`` and threaded through to the
``ExtendedExecutor`` so view-relative transforms (left/right/front/behind)
can resolve their direction at query time.

Convention matches ``maps.sws.geometry``:
- World x is east-positive meters, y is north-positive meters, z is altitude.
- ``yaw_deg`` 0 = facing north (+y), 90 = facing east (+x), increasing
  clockwise viewed from above.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotPose:
    """Drone pose at the moment a CompilerInput is built."""

    x: float  # world east-positive meters
    y: float  # world north-positive meters
    z: float  # altitude meters
    yaw_deg: float
    timestamp: float | None = None
