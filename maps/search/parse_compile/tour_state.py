"""Runtime tour state: the flight-side reader of ``tour.json`` + cursor.

``tour_synth`` owns every ordering decision and writes the result as
``compile/turn_NNNN/tour.json``; this module deserializes that artifact
into ``TourViewpoint`` rows and walks a mutable cursor over them — the
executor (the benchmark sweep, a field GCS loop) advances the cursor
itself.

Two parsers read the same stop-dict schema on purpose:
``FeatureTour.from_tour_json`` here SKIPS malformed stops so a flight
proceeds on whatever is readable, while ``tour_synth.tour_stop_from_dict``
RAISES so a frozen-tour resume never silently drops a stop. Keep both in
mind when the wire schema changes (both hard-fail on the pre-rename
``anchors`` key).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


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

    @classmethod
    def from_tour_json(
        cls,
        workdir: Path | str | None,
        *,
        prefer_latest_turn: bool = True,
    ) -> "FeatureTour":
        """Build the cursor from the synthesizer's ``tour.json``.

        Reads ``<workdir>/compile/turn_NNNN/tour.json`` (latest non-empty
        turn) and seeds the cursor with the synthesized stops. A stop's
        ``prob`` is 0.0 — the synthesizer writes no value-by-stop field.
        Returns an empty tour if no tour artifact is on disk yet.
        """
        if not workdir:
            return cls.empty()
        workdir_path = Path(workdir)
        compile_dir = workdir_path / "compile"
        if not compile_dir.is_dir():
            return cls.empty()
        turn_dirs = sorted(compile_dir.glob("turn_*"))
        if not turn_dirs:
            return cls.empty()
        candidates = list(reversed(turn_dirs)) if prefer_latest_turn else turn_dirs
        for turn_dir in candidates:
            tour_path = turn_dir / "tour.json"
            if not tour_path.is_file():
                continue
            try:
                payload = json.loads(tour_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("tour_state: %s unreadable: %s", tour_path, exc)
                continue
            stops = payload.get("stops") or []
            viewpoints: list[TourViewpoint] = []
            for stop in stops:
                if not isinstance(stop, dict):
                    continue
                pose = stop.get("pose") or {}
                x_raw = pose.get("x")
                y_raw = pose.get("y")
                if x_raw is None or y_raw is None:
                    continue
                try:
                    x = float(x_raw)
                    y = float(y_raw)
                except (TypeError, ValueError):
                    continue
                alt_raw = pose.get("alt_m")
                try:
                    alt_m: float | None = (
                        float(alt_raw) if alt_raw is not None else None
                    )
                except (TypeError, ValueError):
                    alt_m = None
                hdg_raw = pose.get("heading_deg")
                try:
                    heading_deg: float | None = (
                        float(hdg_raw) if hdg_raw is not None else None
                    )
                except (TypeError, ValueError):
                    heading_deg = None
                fid = str(stop.get("stop_id") or "?")
                # ``prob`` is unconditionally 0.0 for a from_tour_json stop:
                # the synthesizer's ``tour_stop_to_dict`` writes no
                # value-by-stop field, so there is nothing to carry through.
                prob = 0.0
                if "aim_targets" not in stop and "anchors" in stop:
                    raise RuntimeError(
                        "pre-rename tour.json artifact: stop carries the "
                        "legacy 'anchors' wire key; this code reads "
                        "'aim_targets'. Re-bake the tour with current code."
                    )
                aim_payload = stop.get("aim_targets") or []
                aim_targets: list[dict] = []
                if isinstance(aim_payload, list):
                    for entry in aim_payload:
                        if not isinstance(entry, dict):
                            continue
                        afid = entry.get("fid")
                        if not isinstance(afid, str):
                            continue
                        aim_targets.append(
                            {
                                "fid": str(afid),
                                "x": float(entry.get("x", 0.0)),
                                "y": float(entry.get("y", 0.0)),
                                "z": float(entry.get("z", 0.0)),
                                "priority": float(entry.get("priority", 0.0)),
                            }
                        )
                viewpoints.append(
                    TourViewpoint(
                        fid=fid,
                        xy=(x, y),
                        prob=prob,
                        aim_targets=tuple(aim_targets),
                        altitude_m=alt_m,
                        heading_deg=heading_deg,
                    )
                )
            if viewpoints:
                log.info(
                    "tour_state: loaded %d viewpoints from %s",
                    len(viewpoints),
                    tour_path,
                )
                return cls(viewpoints=viewpoints)
        return cls.empty()
