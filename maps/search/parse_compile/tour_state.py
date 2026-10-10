"""Runtime tour state: the flown viewpoints as ``TourViewpoint`` rows and a
cursor over them, advanced by the executor."""

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
    that feature's world centroid in tile UTM frame; ``prob`` is the
    candidate prior (0.0 for anchor-only feature_ids).
    """

    fid: str
    xy: tuple[float, float]
    prob: float
    # Per-stop altitude (m AGL); ``None`` falls back to ``cfg.altitude_m``.
    altitude_m: float | None = None
    # Per-stop drone heading (deg); ``None`` falls back to the
    # transit-bearing heading.
    heading_deg: float | None = None


@dataclass
class FeatureTour:
    """Mutable cursor over an ordered ``TourViewpoint`` list."""

    viewpoints: list[TourViewpoint] = field(default_factory=list)
    cursor: int = 0

    @classmethod
    def empty(cls) -> "FeatureTour":
        return cls(viewpoints=[])
