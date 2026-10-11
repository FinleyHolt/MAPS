"""Runtime tour state: the flown viewpoints as ``TourViewpoint`` rows."""

from __future__ import annotations

from dataclasses import dataclass, field

#: The two non-feature tokens a viewpoint target can carry: the drone's
#: current position and the marked region.
VIEW_POSITION_CURRENT = "current_position"
VIEW_POSITION_MARKER = "marked_region"


@dataclass(frozen=True)
class TourViewpoint:
    """One waypoint on the feature tour.

    ``fid`` is the prebake feature the viewpoint sits over; ``xy`` is
    that feature's world centroid in tile UTM frame.
    """

    fid: str
    xy: tuple[float, float]
    # Per-stop altitude (m AGL); ``None`` falls back to ``cfg.altitude_m``.
    altitude_m: float | None = None


@dataclass
class FeatureTour:
    """The ordered ``TourViewpoint`` list of the stops flown so far."""

    viewpoints: list[TourViewpoint] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "FeatureTour":
        return cls(viewpoints=[])
