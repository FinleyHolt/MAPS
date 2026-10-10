"""Action-space models: how the drone gets from where it is to where it is going.

An aircraft's transit is a property of the aircraft. A quadcopter hovers and
turns in place; a fixed wing cannot. So the model that turns "go to this point"
into a pose stream is declared in config (``flight_model.kind``) and resolved
here through :func:`resolve_flight_model`, and a new airframe is a new entry in
``_FLIGHT_MODELS`` rather than an edit to the runner.

**A model is anything with three methods.** No base class, because the
interface is small and already fixed by its one consumer
(``run_benchmark/trajectory.py``):

    set_next_target(xy, *, altitude_m=None)   aim at a point; replaces any
                                              pending route
    has_pending_target() -> bool              is there route left to fly
    __next__() -> FlightPose                  fly ONE tick of ``dt_s``

plus two numbers the replay needs to reason about a leg before flying it:

    arrival_tol_m        how close to a target this model can actually stop
    advance_per_tick_m   how far one tick carries the drone

These are different questions and must not be conflated. A discrete walk stops
within half its 5 m step, so the replay aims half a step past a commit to
centre the landing; a continuous model flies a partial last tick and stops ON
the point, so it aims past by nothing. Its ``advance_per_tick_m`` is still a
whole tick, because that one sizes the commit budget reserve. Reading both off
the model keeps one airframe's numbers out of another's -- a literal 5 m in
the replay would size a jet's reserve in quadcopter steps, and a literal 2.5 m
would land every jet past its target.

``discrete6`` is the CityNav six-action contract (``move_forward 5 m``,
``turn_left 30 deg``, ``turn_right 30 deg``, ``ascend 2 m``, ``descend 2 m``,
``stop``). What gets submitted is the POSE sequence (``visited_xy``); the
alphabet is what makes it submittable, because every pose in it is reachable
by the paper's executor. ``discrete6`` is permanently one of the models, never
the one that gets replaced.

Kept robot-agnostic: no ROS, no perception, no benchmark-state imports. Every
airframe's intrinsics arrive as numbers from config, never as an import of
``robots/``.

Heading convention matches ``EpisodeState.heading_deg`` in the runner:
0 deg = +Y (north), clockwise-positive, degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from benchmarks.citynav.action_snap import (
    FORWARD_STEP_M,
    SNAP_XY_TOL_M,
    CityNavAction,
    Pose,
    apply_action,
    snap_to_actions,
)


@dataclass
class FlightModelConfig:
    """Which action-space model flies, and its intrinsics.

    ``kind`` selects the model (see :func:`known_flight_kinds`).
    ``camera_hz`` sets the per-tick timestamp spacing (``dt_s``) on the
    emitted ``FlightPose`` stream, and is what the offline efficiency reader
    turns frames into seconds with. ``speed_mps`` is the cruise speed of the
    continuous models; ``discrete6`` has no speed, because its 5 m step per
    tick is fixed by the CityNav protocol.

    An intrinsic that the declared ``kind`` does not fly is REFUSED rather
    than ignored: a speed sitting unread in a config reads as a setting that
    is in force.
    """

    camera_hz: float = 5.0
    kind: str = "discrete6"
    speed_mps: float | None = None

    def __post_init__(self) -> None:
        if self.camera_hz <= 0.0:
            raise ValueError("camera_hz must be > 0")
        if self.kind not in _FLIGHT_MODELS:
            raise ValueError(
                f"unknown flight_model.kind {self.kind!r} "
                f"(have: {', '.join(known_flight_kinds())})"
            )
        needs_speed = self.kind in _SPEED_DRIVEN_KINDS
        if needs_speed and not (self.speed_mps and self.speed_mps > 0.0):
            raise ValueError(
                f"flight_model.kind {self.kind!r} flies at a speed: "
                f"set speed_mps > 0"
            )
        if not needs_speed and self.speed_mps is not None:
            raise ValueError(
                f"flight_model.kind {self.kind!r} has no speed_mps -- its "
                f"{FORWARD_STEP_M} m step per tick is fixed by the CityNav "
                f"protocol, so the value would go unread"
            )

    @property
    def dt_s(self) -> float:
        return 1.0 / self.camera_hz


@dataclass(frozen=True)
class FlightPose:
    x: float
    y: float
    heading_deg: float
    frame_idx: int
    t_s: float
    # ``altitude_m`` is the pose AGL after this tick. ``action`` names the
    # CityNav primitive that produced it, or is None for a model with no
    # discrete alphabet. Nothing persists ``action`` today -- the submitted
    # artifact is the pose sequence -- so its readers are the tests that pin
    # the alphabet and pin that replaying the actions reproduces the poses.
    altitude_m: float | None = None
    action: str | None = None


class DiscreteFlightSimulator:
    """The CityNav six-action contract: one queued action per tick.

    Each ``__next__`` consumes one queued ``CityNavAction`` and yields a
    ``FlightPose`` carrying the post-action xy, altitude, heading, plus
    the emitted action name.

    ``set_next_target`` invokes ``snap_to_actions`` against the current pose to
    enqueue the action sequence that drives the drone to the next point.
    Optional ``altitude_m`` lets the caller request altitude changes that emit
    ``ascend`` / ``descend`` actions; omitted altitude keeps the current AGL.
    An exhausted queue yields ``stop`` actions -- the drone hovers rather than
    ending the stream, since hovering is a thing this airframe can do.
    """

    #: The discrete walk stops within about half a 5 m step of what it aimed at.
    arrival_tol_m = SNAP_XY_TOL_M
    #: One ``move_forward``. Turns and altitude changes cover no ground, so
    #: this is the ceiling on a tick's progress, which is what sizing wants.
    advance_per_tick_m = FORWARD_STEP_M

    def __init__(
        self,
        cfg: FlightModelConfig,
        start_pose: tuple[float, float, float],
        *,
        start_altitude_m: float = 0.0,
    ) -> None:
        self._cfg = cfg
        self._pose = Pose(
            x=float(start_pose[0]),
            y=float(start_pose[1]),
            altitude_m=float(start_altitude_m),
            heading_deg=float(start_pose[2]) % 360.0,
        )
        self._action_queue: list[CityNavAction] = []
        self._dynamic_mode = False
        self._done = False
        self._frame_idx = 0

    def _snap(self, target_xy: tuple[float, float], altitude_m: float | None) -> None:
        target = Pose(
            x=float(target_xy[0]),
            y=float(target_xy[1]),
            altitude_m=(
                float(altitude_m) if altitude_m is not None else self._pose.altitude_m
            ),
            heading_deg=self._pose.heading_deg,
        )
        actions, _achieved = snap_to_actions(self._pose, target)
        self._action_queue.extend(actions)

    def set_next_target(
        self,
        xy: tuple[float, float],
        *,
        altitude_m: float | None = None,
    ) -> None:
        """Replace any pending action queue with the snap to ``xy``."""
        self._action_queue = []
        self._snap(xy, altitude_m)
        self._dynamic_mode = True
        self._done = False

    def has_pending_target(self) -> bool:
        return bool(self._action_queue)

    def __iter__(self) -> Iterator[FlightPose]:
        return self

    def __next__(self) -> FlightPose:
        if self._done:
            raise StopIteration
        if self._action_queue:
            action = self._action_queue.pop(0)
        elif self._dynamic_mode:
            action = CityNavAction.STOP
        else:
            self._done = True
            action = CityNavAction.STOP
        self._pose = apply_action(self._pose, action)
        out = FlightPose(
            x=self._pose.x,
            y=self._pose.y,
            heading_deg=self._pose.heading_deg,
            frame_idx=self._frame_idx,
            t_s=self._frame_idx * self._cfg.dt_s,
            altitude_m=self._pose.altitude_m,
            action=action.value,
        )
        self._frame_idx += 1
        return out


def _straight(cfg, start_pose, *, start_altitude_m):
    # Imported here, not at module top: ``straight_flight`` needs
    # ``FlightPose`` from this module, so a top-level import would close a
    # cycle. A model's own file may depend on the registry's; not the reverse.
    from benchmarks.citynav.straight_flight import StraightFlightSimulator

    return StraightFlightSimulator(cfg, start_pose, start_altitude_m=start_altitude_m)


# The registry. A config's ``flight_model.kind`` names one of these; the
# factory takes the config plus the episode's start pose and returns a model.
# Adding an airframe means adding a line here and its intrinsics to the config
# block -- never an edit to the replay, and never an import of ``robots/``.
_FLIGHT_MODELS = {
    "discrete6": DiscreteFlightSimulator,
    "straight": _straight,
}

#: Kinds whose transit is priced by a cruise speed rather than a fixed step.
_SPEED_DRIVEN_KINDS = frozenset({"straight"})


def known_flight_kinds() -> tuple[str, ...]:
    """Every ``flight_model.kind`` this build can fly, for error text."""
    return tuple(sorted(_FLIGHT_MODELS))


def resolve_flight_model(
    cfg: FlightModelConfig,
    *,
    start_pose: tuple[float, float, float],
    start_altitude_m: float,
):
    """The action-space model ``cfg`` declares, positioned at the start pose.

    ``FlightModelConfig.__post_init__`` already refused an unknown kind at
    config load, which is where a typo should stop -- not one leg into a
    replay.
    """
    return _FLIGHT_MODELS[cfg.kind](
        cfg, start_pose, start_altitude_m=start_altitude_m
    )
