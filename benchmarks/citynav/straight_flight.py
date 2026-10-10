"""Straight-line transit at an aircraft speed: the cheapest honest flight.

The drone points at the target and flies to it, covering ``speed_mps * dt_s``
per tick. No quantization, no turn actions, no hover-and-rotate -- the six
discrete primitives are a benchmark protocol, not a thing an aircraft does, and
a straight leg is what any airframe would actually fly given no constraints.

That makes this the FLOOR for the airframe models: every constrained model
(a turn-radius-limited fixed wing, a jet) flies at least as far as this for the
same stop sequence, so the path length it reports is the yardstick their extra
cost is read against.

Deliberately not modelled: acceleration, wind, and any turn constraint. The
first two would move time of flight without moving path length or the score,
and the third is exactly what the constrained models exist to add. Adding them
here would make this an aircraft rather than a floor.

Heading convention matches the rest of the tree: 0 deg = +Y (north),
clockwise-positive.
"""

from __future__ import annotations

import math

from benchmarks.citynav.action_snap import bearing_deg
from benchmarks.citynav.flight_model import FlightModelConfig, FlightPose

# A continuous model lands ON the point: the last tick of a leg is a partial
# one, so there is no quantum to stop short by. This is float dust, not a
# tolerance -- it exists because "arrived" cannot be spelled `== 0.0`.
ARRIVAL_EPS_M = 1e-6


class StraightFlightSimulator:
    """Constant-speed straight-line transit between points.

    One ``__next__`` advances one ``dt_s`` tick toward the current target and
    yields the pose it reached. Altitude ramps linearly across the leg so the
    climb finishes when the traverse does -- the leg is the unit, and an
    airframe with no hover has nowhere to wait out a separate climb.

    An exhausted route holds position: the same liveness contract the discrete
    model has, so the replay's drain loop reads identically for both.
    """

    def __init__(
        self,
        cfg: FlightModelConfig,
        start_pose: tuple[float, float, float],
        *,
        start_altitude_m: float = 0.0,
    ) -> None:
        self._cfg = cfg
        self._step_m = float(cfg.speed_mps) * cfg.dt_s
        self._x = float(start_pose[0])
        self._y = float(start_pose[1])
        self._heading_deg = float(start_pose[2]) % 360.0
        self._alt = float(start_altitude_m)
        self._target: tuple[float, float] | None = None
        self._target_alt: float | None = None
        self._alt_rate: float = 0.0
        self._frame_idx = 0

    @property
    def arrival_tol_m(self) -> float:
        """Effectively zero: the last tick of a leg is partial, so this model
        stops ON the target rather than within some quantum of it.

        It must NOT be one tick. ``arrival_tol_m`` is what the replay aims
        PAST the commit by, to centre an approach that stops short (the
        discrete walk's 2.5 m is half its 5 m step). Reporting a whole tick
        here made every straight commit land uniformly PAST the target --
        5.9 m mean at the C-12's 11.3 m tick, against a 20 m success radius
        whose knife edge the replay itself is written around.
        """
        return ARRIVAL_EPS_M

    @property
    def advance_per_tick_m(self) -> float:
        return self._step_m

    def _remaining_m(self) -> float:
        if self._target is None:
            return 0.0
        return math.hypot(self._target[0] - self._x, self._target[1] - self._y)

    def set_next_target(
        self,
        xy: tuple[float, float],
        *,
        altitude_m: float | None = None,
    ) -> None:
        """Aim at ``xy``, replacing any leg in progress."""
        self._target = (float(xy[0]), float(xy[1]))
        self._target_alt = float(altitude_m) if altitude_m is not None else None
        # Spread the altitude change over the ticks the traverse will take, so
        # the drone arrives level rather than climbing on the spot.
        self._alt_rate = 0.0
        if self._target_alt is not None:
            ticks = max(1.0, math.ceil(self._remaining_m() / self._step_m))
            self._alt_rate = (self._target_alt - self._alt) / ticks

    def has_pending_target(self) -> bool:
        """Pending until the drone is ON the target, not within one tick of it.

        The final tick of a leg is a partial one (``travel = min(step,
        remaining)``), so stopping at ``remaining <= step`` throws away the
        tick that actually arrives. It also meant a commit closer than one
        tick never moved the drone at all -- and the commit leg is the one
        that sets ``pred_xy``, so that was a scored position left at the
        previous pose.
        """
        if self._target is None:
            return False
        if self._remaining_m() > ARRIVAL_EPS_M:
            return True
        # Arrived in xy; a pure climb or descent still has ticks to fly.
        return (
            self._target_alt is not None
            and abs(self._target_alt - self._alt) > 1e-9
        )

    def __iter__(self) -> "StraightFlightSimulator":
        return self

    def __next__(self) -> FlightPose:
        if self._target is not None:
            remaining = self._remaining_m()
            if remaining > 1e-12:
                self._heading_deg = bearing_deg((self._x, self._y), self._target) % 360.0
                travel = min(self._step_m, remaining)
                self._x += travel * (self._target[0] - self._x) / remaining
                self._y += travel * (self._target[1] - self._y) / remaining
            if self._target_alt is not None:
                if abs(self._target_alt - self._alt) <= abs(self._alt_rate):
                    self._alt = self._target_alt
                else:
                    self._alt += self._alt_rate
        out = FlightPose(
            x=self._x,
            y=self._y,
            heading_deg=self._heading_deg,
            frame_idx=self._frame_idx,
            t_s=self._frame_idx * self._cfg.dt_s,
            altitude_m=self._alt,
            # No discrete alphabet: a continuous tick is not one of the six
            # primitives, and naming it one would assert this trajectory is
            # replayable by the paper's executor, which it is not.
            action=None,
        )
        self._frame_idx += 1
        return out
