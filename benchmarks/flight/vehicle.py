"""The aircraft as the flight harness sees it: legs go out, poses and recorded frames come back.

On the C-12 the station speaks to the recorder and the cockpit display over
HTTP; these are the semantics it reads off them, and nothing else. A leg is posted
for an area and flown however the vehicle flies it (on the C-12, by the crew
off the cockpit display). The vehicle records frames as it flies, each tagged
with the pose the aircraft had at its stamp (``nav_track.pose_tag``), and
serves the ring's index without pixels and any frame's pixels by sequence.

The vehicle owns the clock the harness waits on: a real aircraft's is the
wall clock, a twin's can run faster than it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from benchmarks.flight.nav_track import Pose


@dataclass(frozen=True)
class Leg:
    """One leg as posted: ``point`` first, then the rest of the pass this stop
    belongs to (``ahead``), all in the tile's local metres. ``area`` is the
    ground the leg is flown for, (lon, lat) samples; ``revision`` counts the
    times a short pass asked it again, and ``note`` says why. The twin flies
    ``point`` and ``ahead``; ``target``, ``step``, ``area``, ``revision`` and
    ``note`` are the hand-off a vehicle for another aircraft shows its crew or
    acts on."""

    target: str
    point: tuple[float, float]
    step: int
    area: list[tuple[float, float]] | None = None
    ahead: list[tuple[float, float]] = field(default_factory=list)
    revision: int = 0
    note: str | None = None


class Vehicle(Protocol):
    #: Why the last ask answered None, or empty.
    last_error: str

    def post(self, leg: Leg) -> None:
        """Hand the aircraft a leg. Never blocks on the aircraft flying it."""

    def pose(self, t: float | None = None) -> Pose | None:
        """The antenna's pose at ``t`` on the vehicle's clock (now by default), or None without a fix."""

    def index(self, since: int | None = None) -> list[dict[str, Any]] | None:
        """Every recorded frame stamped at or after log time ``since``, no
        pixels: ``{"seq", "log_time", "t_host", "pose": <tag>}`` rows. None
        when the recording cannot be asked."""

    def frame(self, seq: int) -> tuple[bytes, float, int, int, dict[str, Any]] | None:
        """``(jpeg, t_host, log_time, seq, tag)`` of one recorded frame, or None."""

    def now(self) -> float:
        """The vehicle's clock, seconds: the axis ``t_host`` and ``pose(t)`` run on."""

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds`` on the vehicle's clock."""
