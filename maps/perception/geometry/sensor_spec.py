"""Sensor parameterization for the perception + planner path.

One dataclass (`SensorSpec`) carries every piece of geometry a downstream
consumer needs: camera intrinsics, mount pitch, range limits, world-fixed
yaw, depth-integration subsample, and body-frame mount offsets. One
registry (`_REGISTRY`) and one constructor (`resolve_sensor`) turn a
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

A run carries one sensor. Every shipped config resolves ``nadir_1008_90``,
the 1008 px sensor the ortho observation is cropped for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from maps.perception.ortho.camera_model import CameraIntrinsics


def mount_rotation_matrix(pitch_deg: float, roll_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Body <- camera rotation for a mount at ``pitch_deg`` (positive-down),
    ``roll_deg`` (body-X / forward-axis roll, positive = right side down) and
    ``yaw_deg`` (about the camera's own optical axis, CW looking out).

    Camera +Z = optical axis, +X = right, +Y = down; body +X = forward,
    +Y = left, +Z = up. The pitch matrix is the canonical axis-fix + Y-pitch;
    the roll is a body-X rotation applied on top. ``roll_deg = 0`` reduces to
    the pure-pitch matrix every existing caller relied on (byte-identical).

    The yaw right-multiplies, so it turns the CAMERA inside the body and not
    the body inside the world. That ordering is the whole point: an installed
    camera's yaw has to compose after the airframe's roll and pitch, and
    folding it into the heading instead tilts the aircraft about axes the
    mount rotated -- which at 180 degrees inverts roll and pitch outright.
    ``yaw_deg = 0`` returns the matrix unchanged.
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
    """Rotation composed between the body frame and the level/heading frame for
    an airframe whose body deviates from level by ``roll_deg`` (about body +X,
    forward) and ``pitch_deg`` (about body +Y, left).

    Body frame: +X = forward, +Y = left, +Z = up (right-handed), matching
    ``mount_rotation_matrix``'s body convention. The perspective projection
    treats the post-mount body vector as level (forward horizontal, +Z true up)
    before applying the heading yaw; when the airframe is banked or pitched that
    assumption is wrong, so the projection left-multiplies this matrix onto the
    body<-camera mount rotation (``R_eff = body_tilt_matrix @ R_bc``) to carry
    the vector into the level frame first. ``roll_deg = pitch_deg = 0`` returns
    the identity, so the static-mount-only path is byte-identical.

    Composed pitch-then-roll (``Rx(roll) @ Ry(pitch)``). The physical sign of
    each angle (which way the correction tilts) is pinned by the flight-video
    overlay validation against known banked frames; forward/inverse projection
    stay exact inverses for any sign because both compose this matrix identically.
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
    Range and body-offset fields carry defaults for a short-range oblique
    stereo mount; the nadir entries override them.
    """

    name: str
    intrinsics: CameraIntrinsics
    mount_pitch_deg: float
    max_range_m: float = 30.0
    min_range_m: float = 0.3
    world_yaw_deg: float = 0.0
    mount_roll_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    depth_subsample: int = 4
    body_offset_x: float = 0.12
    body_offset_y: float = 0.0
    body_offset_z: float = -0.05
    description: str = ""
    """Operator-readable summary of what this rig observes and what it
    does NOT observe. Patched into the reasoner's system prompt so it does
    not request SAM segmentations the geometry cannot confirm. Every entry
    in ``_REGISTRY`` populates this field."""




    def to_dict(self) -> dict[str, Any]:
        """Flat JSON-safe serialization (used in episode.json)."""
        return {
            "name": self.name,
            "hfov_deg": float(self.intrinsics.hfov_deg),
            "vfov_deg": float(self.intrinsics.vfov_deg),
            "image_width": int(self.intrinsics.image_width),
            "image_height": int(self.intrinsics.image_height),
            "mount_pitch_deg": float(self.mount_pitch_deg),
            "max_range_m": float(self.max_range_m),
            "min_range_m": float(self.min_range_m),
            "world_yaw_deg": float(self.world_yaw_deg),
            "mount_yaw_deg": float(self.mount_yaw_deg),
            "mount_roll_deg": float(self.mount_roll_deg),
            "depth_subsample": int(self.depth_subsample),
            "body_offset_x": float(self.body_offset_x),
            "body_offset_y": float(self.body_offset_y),
            "body_offset_z": float(self.body_offset_z),
            "description": str(self.description),
        }


_ARCHER_IMX290_NADIR_DESCRIPTION = (
    "RGB only, 1024x768, 90° HFOV × 73.7° VFOV, pointed straight down: the "
    "Archer's IMX290 rectified to a pinhole. Nadir footprint about 2 x the height "
    "above ground across, 1.5 x along.\n\n"
    "OBSERVABILITY (use this to judge whether absent attributes are "
    "evidence-against vs simply not-surfaceable):\n"
    "  Reliably observable: roof color / shape / material; ground objects "
    "beneath the drone (cars, road markings, athletic field lines); "
    "footprint shape; relative spatial position.\n"
    "  Hard or unreliable: building wall color, facade detail, story count "
    "or vertical extent; text legibility on signage; small entity counts ±1.\n"
    "When an operator description references a hard-to-observe attribute, "
    "do not penalize the verdict for its absence."
)


# Teledyne FLIR Oryx ORX-10GS-32S4C (3.2 MP, 2048x1536, Sony IMX252: a
# 1/1.8" sensor at 3.45 um a pixel, 7.07 x 5.30 mm) behind a 25 mm M112FM25
# C-mount lens: 2 * atan(3.53 / 25) = 16.09 deg across and 12.10 deg down.
# The lens table's 19.9 x 15.0 deg is the same lens on a 2/3" (8.8 x 6.6 mm)
# sensor, which this body is not, and reads a fifth too much ground per
# pixel. The flight measures the pair with no map: `calibration/flow_timing`
# reads the frames' own motion against the receiver's ground speed, and on
# sortie 20260910T162111Z it reads this entry's focal length long by 0.3%
# over 4,140 pairs at 88.8 m/s and 1,034 m AGL (`flow_timing.json`,
# `lens.focal_scale` -0.0028), three metres of terrain at that height, where
# the 2/3" figure reads a fifth. Nadir-mounted on the C-12 belly.
# At the 6,000 ft AGL test band this is a 517 x 388 m footprint at
# 0.25 m/px -- a real recon frame, not the coarse 224 px ortho square.
_ORYX_ORX32_25MM_INTRINSICS = CameraIntrinsics(
    hfov_deg=16.09, vfov_deg=12.10, image_width=2048, image_height=1536
)


_ORYX_ORX32_25MM_NADIR_DESCRIPTION = (
    "RGB only, 2048x1536, 16.1 deg HFOV x 12.1 deg VFOV, pointed "
    "straight down. Narrow telephoto-style nadir FOV: at the 6,000 ft AGL "
    "band the footprint is ~517 x 388 m at ~0.25 m/px, so the frame covers a "
    "few city blocks, not a whole town, and a 5 m object spans ~20 px. Depth "
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
    "~0.5 m (about 2 px at the 6,000 ft band).\n"
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
    # The Archer's IMX290 behind its wide lens, RECTIFIED: the recorder
    # remaps every frame to this pinhole
    # and the calibration solves it at a 90 deg rectified field of view,
    # so the horizontal field of view is 90 deg by
    # construction and the vertical follows the 4:3 frame. What the bench
    # calibration decides is whether 90 deg sits inside the lens's good field.
    "archer_imx290_nadir": SensorSpec(
        name="archer_imx290_nadir",
        intrinsics=CameraIntrinsics(hfov_deg=90.0, vfov_deg=73.7398, image_width=1024, image_height=768),
        mount_pitch_deg=90.0,
        max_range_m=float("inf"),
        min_range_m=0.3,
        # The antenna-to-camera arm is applied by the readers.
        body_offset_x=0.0,
        body_offset_y=0.0,
        body_offset_z=0.0,
        description=_ARCHER_IMX290_NADIR_DESCRIPTION,
    ),
    "nadir_1008_90": SensorSpec(
        name="nadir_1008_90",
        intrinsics=_NADIR_1008_INTRINSICS,
        # Straight down, matching the name and what the runtime ingest
        # requires. Inert on the ortho path (the crop is orthographic and
        # carries no sensor), live on the field DEM path.
        mount_pitch_deg=90.0,
        max_range_m=float("inf"),
        min_range_m=0.3,
        world_yaw_deg=0.0,
        depth_subsample=1,
        body_offset_x=0.0,
        body_offset_y=0.0,
        body_offset_z=0.0,
        description=_NADIR_1008_90_DESCRIPTION,
    ),
    "oryx_orx32_25mm_nadir": SensorSpec(
        name="oryx_orx32_25mm_nadir",
        intrinsics=_ORYX_ORX32_25MM_INTRINSICS,
        mount_pitch_deg=90.0,
        # Depth is DSM-raycast (no stereo); 8 km march cap lives in the
        # renderer, so the sensor envelope is effectively infinite.
        max_range_m=float("inf"),
        min_range_m=0.3,
        # Belly-mounted on the C-12, roughly under the CG, and installed
        # facing AFT: the frame's top is the tail, not the nose. The frames
        # say so with no map at all: the ground moves UP the frame on both
        # headings of sortie 20260910T162111Z (`calibration/flow_timing`,
        # `lens.flow_minus_track_deg` reads 180 on every leg), where a
        # forward mount sees it move down. A heading-sign error would swap
        # which way between the two headings; a mount reads the same on both.
        mount_yaw_deg=180.0,
        body_offset_x=0.0,
        body_offset_y=0.0,
        body_offset_z=-0.5,
        description=_ORYX_ORX32_25MM_NADIR_DESCRIPTION,
    ),
}


def list_sensors() -> list[str]:
    """Return every registered sensor name (for CLI help and error messages)."""
    return sorted(_REGISTRY.keys())


def sensor_catalog() -> list[dict[str, Any]]:
    """Every registered sensor as a flat dict, name-sorted, JSON-safe.

    What a UI needs to offer the rig as a choice rather than compile one in:
    the map readouts derive "the altitude that sees this much ground" from a
    picked sensor's HFOV, and this registry is where HFOV lives. Same
    ``to_dict`` shape the episode record carries, so a served sensor and a
    recorded one are read the same way — except that the depth-less rigs
    carry ``max_range_m = inf``, which ``json.dumps`` writes as a bare
    ``Infinity``. That is not JSON: a browser's ``JSON.parse`` rejects the
    whole document, so a served catalog arrives empty and the readout
    silently loses its field of view. Non-finite becomes ``null`` here,
    which is what "no range envelope" means to a reader anyway.
    """
    return [
        {k: (None if isinstance(v, float) and not math.isfinite(v) else v)
         for k, v in _REGISTRY[name].to_dict().items()}
        for name in list_sensors()
    ]


# What an inline override may name: a registered base and SensorSpec.to_dict's keys.
_INLINE_KEYS = frozenset({"base", *_REGISTRY["nadir_1008_90"].to_dict()})


def resolve_sensor(spec: str | Mapping[str, Any] | None) -> SensorSpec | None:
    """Turn ``cfg.sensor`` into a ``SensorSpec``.

    Accepts:
    * ``None`` or empty -> returns ``None`` (caller should fall back to
      orthographic raster rendering).
    * Registered string name -> looks up the registry.
    * Inline dict with any of ``hfov_deg``, ``vfov_deg``, ``image_width``,
      ``image_height``, ``mount_pitch_deg``, ``max_range_m``, etc. A
      ``base`` key picks a registered starting point; other keys override
      that base.
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
    base_name = spec.get("base")
    base = (
        _REGISTRY[str(base_name)]
        if base_name is not None
        else _REGISTRY["nadir_1008_90"]
    )
    intr = base.intrinsics
    return SensorSpec(
        name=str(spec.get("name", base.name + "_override")),
        intrinsics=CameraIntrinsics(
            hfov_deg=float(spec.get("hfov_deg", intr.hfov_deg)),
            vfov_deg=float(spec.get("vfov_deg", intr.vfov_deg)),
            image_width=int(spec.get("image_width", intr.image_width)),
            image_height=int(spec.get("image_height", intr.image_height)),
        ),
        mount_pitch_deg=float(spec.get("mount_pitch_deg", base.mount_pitch_deg)),
        max_range_m=float(spec.get("max_range_m", base.max_range_m)),
        min_range_m=float(spec.get("min_range_m", base.min_range_m)),
        world_yaw_deg=float(spec.get("world_yaw_deg", base.world_yaw_deg)),
        mount_yaw_deg=float(spec.get("mount_yaw_deg", base.mount_yaw_deg)),
        mount_roll_deg=float(spec.get("mount_roll_deg", base.mount_roll_deg)),
        depth_subsample=int(spec.get("depth_subsample", base.depth_subsample)),
        body_offset_x=float(spec.get("body_offset_x", base.body_offset_x)),
        body_offset_y=float(spec.get("body_offset_y", base.body_offset_y)),
        body_offset_z=float(spec.get("body_offset_z", base.body_offset_z)),
        description=str(spec.get("description", base.description)),
    )
