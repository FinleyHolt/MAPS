"""Runtime tour state: the flown viewpoints as ``TourViewpoint`` rows and a
cursor over them, advanced by the executor."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TourViewpoint:
    """One waypoint on the feature tour.

    ``fid`` is the prebake feature the viewpoint sits over; ``xy`` is
    that feature's world centroid in tile UTM frame; ``prob`` is the
    candidate prior (0.0 for anchor-only feature_ids). ``aim_targets``
    is the ordered sweep over the distinct targets to observe here. Each
    entry carries ``{fid, x, y, z, priority}`` — a place, not a camera
    pose, because the observation is a nadir ortho crop. The runtime walks
    ``aim_targets`` in order, counting ``SEARCH_ANCHOR_DWELL_FRAMES``
    settled frames before moving on.
    """

    fid: str
    xy: tuple[float, float]
    prob: float
    aim_targets: tuple[dict, ...] = ()
    # Per-stop geometric altitude (m AGL) baked by the tour synth.
    # ``None`` falls back to ``cfg.altitude_m`` at flight time; a finite
    # value is teleported / snapped directly so each stop honors the
    # synth's FOV-fit pick (top_down) or pitch-band-centered pick (side).
    altitude_m: float | None = None
    # Per-stop drone heading (deg) the synth picked so the body
    # forward-axis points at the aim target; the runtime applies this on
    # teleport. ``None`` falls back to the transit-bearing heading.
    heading_deg: float | None = None


@dataclass
class FeatureTour:
    """Mutable cursor over an ordered ``TourViewpoint`` list."""

    viewpoints: list[TourViewpoint] = field(default_factory=list)
    cursor: int = 0

    def current(self) -> TourViewpoint | None:
        if 0 <= self.cursor < len(self.viewpoints):
            return self.viewpoints[self.cursor]
        return None


    @classmethod
    def empty(cls) -> "FeatureTour":
        return cls(viewpoints=[])
