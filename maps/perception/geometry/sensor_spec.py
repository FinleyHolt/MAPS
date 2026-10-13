"""Sensor parameterization for the perception + planner path.

One dataclass (`SensorSpec`) carries every piece of geometry a downstream
consumer needs: camera intrinsics, mount angles, range limit and world-fixed
yaw. One registry (`_REGISTRY`) and one constructor (`resolve_sensor`) turn a
string name or inline dict into a fully-populated spec.

Two yaws, and they compose at different points. ``world_yaw_deg`` is the
sensor's world-frame optical-axis bearing (0 = +north, CW-positive); it is 0
for a body-mounted sensor and downstream callers add the live body heading to
it, so a small fitted angle (a boresight yaw, a crab) rides here.
``mount_yaw_deg`` is how the camera is INSTALLED in the airframe, about its own
optical axis: 0 looking forward, 180 bolted in facing aft. It composes after
the airframe's roll and pitch rather than before them, which is what a large
angle needs -- fold 180 into the heading instead and the aircraft gets tilted
about axes the mount turned, inverting roll and pitch. The two are
interchangeable only for a level airframe, which is why the small fitted
angles can keep the cheaper seat.

Pitch convention: positive = looking down; nadir is +90 deg.

A run carries one sensor: the CityNav config resolves ``nadir_1008_90``, the
1008 px sensor its ortho observation is cropped for, and the Cal City config
the C-12's ``oryx_orx32_25mm_nadir``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Mapping

import numpy as np



@dataclass(frozen=True)
class CameraIntrinsics:
    """A plain perspective pinhole: fields of view and pixel size."""

    hfov_deg: float
    vfov_deg: float
    image_width: int
    image_height: int

    @property
    def fx(self) -> float:
        return self.image_width / (2.0 * np.tan(np.deg2rad(self.hfov_deg) / 2.0))

    @property
    def fy(self) -> float:
        return self.image_height / (2.0 * np.tan(np.deg2rad(self.vfov_deg) / 2.0))

    @property
    def cx(self) -> float:
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        return self.image_height / 2.0


def mount_rotation_matrix(pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Body <- camera rotation for a mount at ``pitch_deg`` (positive-down),
    ``roll_deg`` (about body +X, positive = right side down) and ``yaw_deg``
    (about the camera's own optical axis, CW looking out).

    Camera +Z = optical axis, +X = right, +Y = down; body +X = forward,
    +Y = left, +Z = up. The roll left-multiplies the pitch; the yaw
    right-multiplies, turning the camera inside the body (see the module
    docstring on the two yaws).
    """
    pr = math.radians(float(pitch_deg))
    cp, sp = math.cos(pr), math.sin(pr)
    r_pitch = np.array([[0.0, -sp, cp], [-1.0, 0.0, 0.0], [0.0, -cp, -sp]])
    out = r_pitch
    if roll_deg != 0.0:
        rr = math.radians(float(roll_deg))
        cr, sr = math.cos(rr), math.sin(rr)
        out = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]) @ out
    if yaw_deg != 0.0:
        yr = math.radians(float(yaw_deg))
        cy, sy = math.cos(yr), math.sin(yr)
        out = out @ np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return out


def body_tilt_matrix(roll_deg: float, pitch_deg: float) -> np.ndarray:
    """Level frame <- body rotation for an airframe tilted ``roll_deg`` (about
    body +X, forward) and ``pitch_deg`` (about body +Y, left), composed
    ``Rx(roll) @ Ry(pitch)``; the projection left-multiplies it onto the mount
    rotation so the heading yaw acts on a level vector. Identity at 0, 0.
    """
    if roll_deg == 0.0 and pitch_deg == 0.0:
        return np.eye(3)
    rr = math.radians(float(roll_deg))
    pr = math.radians(float(pitch_deg))
    cr, sr = math.cos(rr), math.sin(rr)
    cp, sp = math.cos(pr), math.sin(pr)
    r_roll = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    r_pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return r_roll @ r_pitch


@dataclass(frozen=True)
class SensorSpec:
    """Per-sensor geometry. Immutable.

    ``mount_pitch_deg`` is positive-down in the body frame (+90 = nadir).
    The ``max_range_m`` default is a short-range stereo envelope; both
    registered sensors set it infinite.
    """

    name: str
    intrinsics: CameraIntrinsics
    mount_pitch_deg: float
    max_range_m: float = 30.0
    world_yaw_deg: float = 0.0
    mount_roll_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    description: str = ""
    """Operator-readable summary of what this rig observes and what it
    does NOT observe. Patched into the reasoner's system prompt so it does
    not request SAM segmentations the geometry cannot confirm. Every entry
    in ``_REGISTRY`` populates this field."""


# Teledyne FLIR Oryx ORX-10GS-32S4C (2048x1536, Sony IMX252: a 1/1.8" sensor,
# 7.07 x 5.30 mm) behind a 25 mm M112FM25 C-mount lens: 2 * atan(3.53 / 25) =
# 16.09 deg across and 12.10 deg down. The lens table's 19.9 x 15.0 deg is the
# same lens on a 2/3" sensor, which this body is not. Nadir-mounted on the
# C-12 belly.
_ORYX_ORX32_25MM_INTRINSICS = CameraIntrinsics(
    hfov_deg=16.09, vfov_deg=12.10, image_width=2048, image_height=1536
)


_ORYX_ORX32_25MM_NADIR_DESCRIPTION = (
    "RGB only, 2048x1536, 16.1 deg HFOV x 12.1 deg VFOV, pointed "
    "straight down. Narrow telephoto-style nadir FOV: at the {agl_ft:,.0f} ft "
    "AGL this run flies, the footprint is ~{w:.0f} x {h:.0f} m at ~{gsd:.2f} m/px, "
    "so the frame covers a few city blocks, not a whole town, and a 5 m object "
    "spans ~{px_5m:.0f} px. Depth "
    "is synthesised post-flight by raycasting the terrain surface (no stereo), "
    "so it has no sensor range envelope.\n\n"
    "OBSERVABILITY (use this to judge whether absent attributes are "
    "evidence-against vs simply not-surfaceable):\n"
    "  Reliably observable: roof color / shape / material; runway, taxiway, "
    "apron and ramp markings; aircraft, vehicles and other large ground "
    "objects directly beneath; building and hangar footprint shape and size; "
    "relative spatial position (north-of, south-of, between) of top-down "
    "visible features; parking and storage-yard layout.\n"
    "  Hard or unreliable: building wall color, facades, story count and "
    "vertical extent (the 2.5D DSM does not model walls); text legibility on "
    "signage or tail numbers; small entity counts +-1; objects smaller than "
    "~0.5 m (about {px_half_m:.0f} px at that height).\n"
    "When an operator description references a hard-to-observe attribute, do "
    "not penalize the verdict for its absence."
)


_NADIR_1008_INTRINSICS = CameraIntrinsics(
    hfov_deg=90.0, vfov_deg=90.0, image_width=1008, image_height=1008
)


_NADIR_1008_90_DESCRIPTION = (
    "RGB, 1008×1008 square FOV (90° HFOV × 90° VFOV). The view is "
    "always straight down: an altitude-driven orthographic nadir crop, "
    "north-up.\n\n"
    "OBSERVABILITY (use this to judge whether absent attributes are "
    "evidence-against vs simply not-surfaceable):\n"
    "  Reliable: roof color, roof shape, roof material; ground-level "
    "objects directly beneath (cars, dumpsters, road markings, "
    "footpaths); footprint shape of buildings and parking lots; "
    "top-down spatial relations (north-of, south-of, between).\n"
    "  Unreliable: wall color, facade detail, story count, vertical "
    "extent — nothing vertical faces the camera.\n"
    "  Hard: text legibility on small signs; small entity counts ±1 "
    "(SAM may merge a tight cluster of N objects into one mask or split "
    "one into N+1).\n"
    "When an operator description references an attribute a top-down view "
    "cannot surface, do not penalize the verdict for its absence."
)


_REGISTRY: dict[str, SensorSpec] = {
    "nadir_1008_90": SensorSpec(
        name="nadir_1008_90",
        intrinsics=_NADIR_1008_INTRINSICS,
        # Straight down, matching the name and what the runtime ingest
        # requires. Inert on the ortho path (the crop is orthographic and
        # carries no sensor), live on the field DEM path.
        mount_pitch_deg=90.0,
        max_range_m=float("inf"),
        world_yaw_deg=0.0,
        description=_NADIR_1008_90_DESCRIPTION,
    ),
    "oryx_orx32_25mm_nadir": SensorSpec(
        name="oryx_orx32_25mm_nadir",
        intrinsics=_ORYX_ORX32_25MM_INTRINSICS,
        mount_pitch_deg=90.0,
        # Depth is DSM-raycast (no stereo); 8 km march cap lives in the
        # renderer, so the sensor envelope is effectively infinite.
        max_range_m=float("inf"),
        # Installed facing AFT: the frame's top is the tail. The ground moves
        # up the frame on both headings, where a forward mount sees it move
        # down (a heading-sign error would differ between headings).
        mount_yaw_deg=180.0,
        description=_ORYX_ORX32_25MM_NADIR_DESCRIPTION,
    ),
}


def served_description(spec: SensorSpec, *, agl_m: float | None) -> str:
    """The rig's description as served. A description that states a footprint
    states the sensor's own (``sensor_frame.footprint_m``) at ``agl_m``, the
    height above ground the mission flies; one that states none is served as
    written."""
    if "{" not in spec.description:
        return spec.description
    if agl_m is None:
        raise ValueError(f"{spec.name} states its footprint at the height the mission flies: "
                         "no height above ground was given")
    from maps.perception.ortho.sensor_frame import footprint_m  # noqa: PLC0415

    w, h = footprint_m(spec, agl_m)
    gsd = w / spec.intrinsics.image_width
    return spec.description.format(agl_ft=agl_m / 0.3048, w=w, h=h, gsd=gsd, px_5m=5.0 / gsd,
                                   px_half_m=0.5 / gsd)


def list_sensors() -> list[str]:
    """Return every registered sensor name (for CLI help and error messages)."""
    return sorted(_REGISTRY.keys())


_INTRINSIC_KEYS = tuple(f.name for f in fields(CameraIntrinsics))
# Keys the flown boresight chains carry in their inline sensor that nothing reads.
_IGNORED_KEYS = frozenset({"min_range_m", "depth_subsample", "body_offset_x", "body_offset_y", "body_offset_z"})
# What an inline override may name: a registered base, the intrinsics, and SensorSpec's other fields.
_INLINE_KEYS = frozenset({"base", *_INTRINSIC_KEYS, *(f.name for f in fields(SensorSpec)), *_IGNORED_KEYS}) - {"intrinsics"}


def resolve_sensor(spec: str | Mapping[str, Any] | None) -> SensorSpec | None:
    """``cfg.sensor`` as a ``SensorSpec``: ``None`` for ``None`` or ``""``, the
    registry entry for a name, or for an inline dict its ``base`` entry
    (default ``nadir_1008_90``) with the other keys overriding it, each cast to
    the type the base value carries. An unknown key raises.
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        if spec in _REGISTRY:
            return _REGISTRY[spec]
        raise ValueError(
            f"Unknown sensor {spec!r}. Registered sensors: {list_sensors()}"
        )
    if not isinstance(spec, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(  # pyright: ignore[reportUnreachable]
            f"cfg.sensor must be str, dict, or None; got {type(spec)}"
        )
    unknown = sorted(set(spec) - _INLINE_KEYS)
    if unknown:
        raise ValueError(f"unknown sensor override key(s) {unknown}; known: {sorted(_INLINE_KEYS)}")
    base_name = "nadir_1008_90" if spec.get("base") is None else spec["base"]
    if not isinstance(base_name, str) or base_name not in _REGISTRY:
        raise ValueError(f"sensor base {base_name!r} is not registered: {list_sensors()}")
    base = _REGISTRY[base_name]
    over = {}
    for k, v in spec.items():
        if k not in ("base", *_IGNORED_KEYS):
            held = base.intrinsics if k in _INTRINSIC_KEYS else base
            want = type(getattr(held, k))  # the type the base value carries
            try:
                if not isinstance(v, (str, int, float)):
                    raise TypeError
                over[k] = want(v)
            except (TypeError, ValueError):
                raise ValueError(f"sensor {k} must be a {want.__name__}, got {v!r}") from None
    intr = replace(base.intrinsics, **{k: over.pop(k) for k in _INTRINSIC_KEYS if k in over})
    over.setdefault("name", base.name + "_override")
    return replace(base, intrinsics=intr, **over)
