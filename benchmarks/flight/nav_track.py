"""A frame's pose: the tag the recorder keeps beside each frame, and the camera it places.

The recorder reads the aircraft's track at each frame's stamp and keeps the
answer beside the frame (:func:`pose_tag`): the fix and the attitude the
aircraft had at that instant, so a frame is posed at its own stamp by its own
tag. :func:`camera_pose_local` puts that antenna pose in the projector's frame:
the tile's local metres, the camera's position through the airframe's lever
arm, and the camera's altitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from benchmarks.flight.tile_georef import lonlat_to_local


@dataclass(frozen=True)
class Pose:
    """Where the ANTENNA was at ``t``, and how the aircraft was pointed."""

    t: float
    lat: float
    lon: float
    alt_msl_m: float
    alt_hae_m: float
    track_deg: float
    speed_mps: float
    h_acc_m: float
    #: Distance in time to the nearer fix this pose stands on.
    gap_s: float
    utc: str
    heading_deg: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    #: ``fusion`` off a fused attitude, ``estimated`` when the roll is a
    #: coordinated turn's bank off the track's rate, ``track`` otherwise.
    attitude: str = "track"
    climb_mps: float = 0.0


def pose_tag(pose: Pose) -> dict[str, Any]:
    """The pose as the tag a frame carries."""
    return {"lat": pose.lat, "lon": pose.lon, "alt_m": pose.alt_hae_m, "alt_msl_m": pose.alt_msl_m,
            "heading_deg": pose.heading_deg, "roll_deg": pose.roll_deg, "pitch_deg": pose.pitch_deg,
            "attitude": pose.attitude, "track_deg": pose.track_deg, "speed_mps": pose.speed_mps,
            "climb_mps": pose.climb_mps, "h_acc_m": pose.h_acc_m, "utc": pose.utc}


def tag_pose(tag: dict[str, Any] | None, t: float) -> Pose | None:
    """The pose a frame's tag states, at the frame's own stamp ``t``; None
    when the tag names no position or no heading. A tag that names no
    attitude is read as the recorder's fused word."""
    if not tag or tag.get("lat") is None or tag.get("lon") is None or tag.get("heading_deg") is None:
        return None
    alt_hae = float(tag.get("alt_m") if tag.get("alt_m") is not None else tag.get("alt_msl_m", 0.0))
    alt_msl = float(tag.get("alt_msl_m") if tag.get("alt_msl_m") is not None else alt_hae)
    heading = float(tag["heading_deg"])
    return Pose(t=t, lat=float(tag["lat"]), lon=float(tag["lon"]), alt_msl_m=alt_msl, alt_hae_m=alt_hae,
                track_deg=float(tag["track_deg"]) if tag.get("track_deg") is not None else heading,
                speed_mps=float(tag.get("speed_mps") or 0.0),
                h_acc_m=float(tag["h_acc_m"]) if tag.get("h_acc_m") is not None else float("nan"),
                gap_s=0.0, utc=str(tag.get("utc") or ""), heading_deg=heading,
                roll_deg=float(tag.get("roll_deg") or 0.0), pitch_deg=float(tag.get("pitch_deg") or 0.0),
                attitude=str(tag.get("attitude") or "fusion"), climb_mps=float(tag.get("climb_mps") or 0.0))


def camera_pose_local(pose: Pose, tiles_dir, map_name: str, *, lever: tuple[float, float, float]
                      ) -> tuple[tuple[float, float], float, float]:
    """``(pose_xy, alt_msl_m, yaw_deg)`` of the CAMERA in the tile frame.

    ``lever`` is the antenna-to-camera arm ``(fwd, right, down)`` in body
    metres, applied through the aircraft's attitude: the arm rolls and
    pitches with the airframe. Raises when the bake carries no georef; a fix
    that cannot be placed must not be guessed at.
    """
    local = lonlat_to_local(tiles_dir, map_name, pose.lon, pose.lat)
    if local is None:
        raise ValueError(f"{map_name} carries no georef under {tiles_dir}; a fix cannot be placed in the tile frame")
    east, north, down = body_to_ned(tuple(lever), roll_deg=pose.roll_deg, pitch_deg=pose.pitch_deg,
                                    heading_deg=pose.heading_deg)
    return (local[0] + east, local[1] + north), pose.alt_msl_m - down, pose.heading_deg


def body_to_ned(arm: tuple[float, float, float], *, roll_deg: float, pitch_deg: float,
                heading_deg: float) -> tuple[float, float, float]:
    """A body vector ``(fwd, right, down)`` as ``(east, north, down)``:
    ``Rz(psi) Ry(theta) Rx(phi)``, roll right wing down and pitch nose up positive."""
    fwd, right, down = (float(v) for v in arm)
    phi, theta, psi = (math.radians(roll_deg), math.radians(pitch_deg), math.radians(heading_deg))
    cphi, sphi = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    cps, sps = math.cos(psi), math.sin(psi)
    north = fwd * cth * cps + right * (sphi * sth * cps - cphi * sps) + down * (cphi * sth * cps + sphi * sps)
    east = fwd * cth * sps + right * (sphi * sth * sps + cphi * cps) + down * (cphi * sth * sps - sphi * cps)
    ned_down = -fwd * sth + right * sphi * cth + down * cphi * cth
    return east, north, ned_down
