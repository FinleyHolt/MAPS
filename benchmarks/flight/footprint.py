"""Where a nadir camera's frame falls on flat ground, off a pose and no pixels.

The arrival rule and the rank a pass's frames are ordered by both read this:
``off_nadir_deg`` is the angle between the optical axis and the ray to a
ground point, None when the frame does not hold it.
"""

from __future__ import annotations

import math

#: Metres per degree of latitude: the flat-earth step off a fix.
M_PER_DEG_LAT = 111_320.0


def off_nadir_deg(
    east_m: float,
    north_m: float,
    agl_m: float,
    heading_deg: float,
    roll_deg: float,
    pitch_deg: float,
    *,
    hfov_deg: float,
    vfov_deg: float,
) -> float | None:
    """The angle, degrees, between the optical axis and the ray to the ground
    point ``east_m, north_m`` from the camera's ground position, for a nadir
    camera ``agl_m`` up on ``heading_deg`` (0 north, clockwise), banked
    ``roll_deg`` (right wing down positive) and pitched ``pitch_deg`` (nose up
    positive). ``hfov_deg`` spans across track, ``vfov_deg`` along it. None
    when the frame does not hold the point.

    The angle ranks the frames of one pass: an attitude error reaches the
    ground through a lever that is zero at nadir and grows with it, so a
    banked frame is ordered behind a level one that held the same point,
    never refused for the bank.
    """
    if agl_m <= 0.0:
        return None
    h = math.radians(heading_deg)
    # The level body frame: +X forward, +Y left, +Z up; the point sits agl_m below.
    bx = math.sin(h) * east_m + math.cos(h) * north_m
    by = -(math.cos(h) * east_m - math.sin(h) * north_m)
    bz = -agl_m
    # Into the tilted body: the transpose of the projector's Rx(roll) @ Ry(pitch),
    # the pitch in the projector's nose-down sign.
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    cp, sp = math.cos(math.radians(-pitch_deg)), math.sin(math.radians(-pitch_deg))
    x1, y1, z1 = bx, cr * by + sr * bz, -sr * by + cr * bz
    x2, y2, z2 = cp * x1 - sp * z1, y1, sp * x1 + cp * z1
    # The nadir mount: camera +X right (-body Y), +Y down the image (-body X), +Z the optical axis (-body Z).
    cam_x, cam_y, cam_z = -y2, -x2, -z2
    if cam_z <= 0.0:
        return None
    x, y = cam_x / cam_z, cam_y / cam_z
    if abs(x) > math.tan(math.radians(hfov_deg) / 2.0) or abs(y) > math.tan(
        math.radians(vfov_deg) / 2.0
    ):
        return None
    return math.degrees(math.atan(math.hypot(x, y)))


def in_frame_xy(
    pose_xy,
    agl_m: float,
    heading_deg: float,
    roll_deg: float,
    pitch_deg: float,
    target_xy,
    *,
    hfov_deg: float,
    vfov_deg: float,
) -> bool:
    """Whether a camera over ``pose_xy`` holds ``target_xy``, both in local metres (x east, y north)."""
    return (
        off_nadir_deg(
            float(target_xy[0]) - float(pose_xy[0]),
            float(target_xy[1]) - float(pose_xy[1]),
            agl_m,
            heading_deg,
            roll_deg,
            pitch_deg,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
        )
        is not None
    )


def off_nadir_lonlat(
    lat: float,
    lon: float,
    agl_m: float,
    heading_deg: float,
    roll_deg: float,
    pitch_deg: float,
    point_lat: float,
    point_lon: float,
    *,
    hfov_deg: float,
    vfov_deg: float,
) -> float | None:
    """:func:`off_nadir_deg` for a camera over ``lat, lon`` and a point
    ``point_lat, point_lon``: equirectangular offsets, a centimetre over a frame's reach."""
    scale = M_PER_DEG_LAT * math.cos(math.radians(lat))
    return off_nadir_deg(
        (float(point_lon) - float(lon)) * scale,
        (float(point_lat) - float(lat)) * M_PER_DEG_LAT,
        agl_m,
        heading_deg,
        roll_deg,
        pitch_deg,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
    )
