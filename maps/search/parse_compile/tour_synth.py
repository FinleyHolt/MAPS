"""Tour synthesizer: turn the reasoner's viewpoints into a fixed visit list.

:func:`build_initial_tour_from_viewpoints` collapses the reasoner's
viewpoints onto unique stops for the preflight tour;
:func:`synthesize_extension` appends stops mid-flight when the reasoner
asks for more. Every stop is visited; there is no prize, no skipping,
no budget.

Per-stop pose is picked by ``_pick_observation_pose`` ->
``perception.viewpoint.pick_viewpoint``: a nadir, target-centroid
overhead pose at the in-band altitude, with the view square sized as
``2 * (altitude - GROUND_LEVEL[map])``. The picker diag lands on each
``TourStop`` as ``pose_picker_diag`` and persists into ``tour.json``.
The ortho observation looks straight down, so a stop aims at places, not
camera poses.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np

from maps.sws.spatial import grid_to_world_xy
from maps.util.xy_token import parse_xy_token

log = logging.getLogger(__name__)

TOUR_SCHEMA_VERSION = "1.0"
TOUR_SYNTHESIZER_VERSION = "1.0"

# Grounding-free view-position sentinels. A requested viewpoint whose
# ``target_fid`` is one of these needs NO grounded prior node: the compiler
# resolves it to a concrete ``target_pose`` [x, y] from admissible inputs
# (``current_position`` -> the drone/start pose ``inp.robot_pose``;
# ``marked_region`` -> the operator marker ``inp.fallback_marker_xy``, ~25 m
# from the target). ``synthesize_extension`` then drives the pose straight
# off ``target_pose`` (see ``_build_viewpoint_request``), yielding a nadir
# orthographic stop the runtime perceives from. This is the model's
# always-available escape hatch when it cannot bind an anchor.
VIEW_POSITION_CURRENT = "current_position"
VIEW_POSITION_MARKER = "marked_region"
VIEW_POSITION_SENTINELS = frozenset({VIEW_POSITION_CURRENT, VIEW_POSITION_MARKER})


@dataclass
class TourSynthesizerConfig:
    """Knobs that govern per-observation pose resolution, the aim-target
    sweep, and the shortest-path walk over the resulting tour stops.
    """

    fov_m: float = 120.0
    # CityNav-compliant agent action space (defaults from
    # vlnce/actions.py in the official CityNav repo). ``_snap_to_lattice``
    # quantizes every resolved pose onto this grid: MOVE_FORWARD 5 m,
    # TURN ±30°, GO_UP/DOWN ±2 m.
    agent_forward_step_m: float = 5.0
    agent_yaw_step_deg: float = 30.0
    agent_altitude_step_m: float = 2.0

    # Fallback altitude hint (meters) for a viewpoint request when the caller
    # supplies no per-episode start altitude. The orthographic picker sizes its
    # view square from ``altitude_hint_m − ground_z``.
    default_altitude_hint_m: float = 80.0


ViewIntent = Literal["top_down", "side", "any"]
ViewFace = Literal["north", "south", "east", "west", "principal", "any"]


# Cardinal bearings (degrees, north=0, increasing clockwise). The
# resolver uses the bearing from viewpoint toward the candidate
# centroid; "north" face means the viewpoint sits NORTH of the
# candidate and looks south back at it.
_VIEW_FACE_BEARING_DEG: dict[str, float] = {
    "north": 0.0,
    "east": 90.0,
    "south": 180.0,
    "west": 270.0,
}


def rasterise_height_map(runtime_graph: Any, grid_shape: tuple[int, int]) -> np.ndarray:
    """``(H, W) float32`` height-above-ground stamped from the graph.

    Iterates every ``MapObject`` on ``runtime_graph.nodes``; reads its
    ``tags["height_m"]`` (or ``tags["height"]``); rasterizes that height
    onto the node's ``cells`` (``(N, 2) int32`` row/col list) via a
    per-cell ``max`` so taller buildings dominate overlapping
    footprints. Nodes with no height tag or no cells are skipped.

    Used by the visibility-driven pose resolver to threshold an
    obstacle grid at a flight altitude. Pure NumPy, no GPU.
    """
    H, W = int(grid_shape[0]), int(grid_shape[1])
    height_map = np.zeros((H, W), dtype=np.float32)
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return height_map
    for node in nodes.values():
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        raw = tags.get("height_m")
        if raw is None:
            raw = tags.get("height")
        if raw is None:
            continue
        try:
            h = float(raw)
        except (TypeError, ValueError):
            continue
        if not (h > 0.0):
            continue
        cells = getattr(node, "cells", None)
        if cells is None:
            continue
        try:
            cells_arr = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if cells_arr.size == 0:
            continue
        rs = cells_arr[:, 0]
        cs = cells_arr[:, 1]
        in_bounds = (rs >= 0) & (rs < H) & (cs >= 0) & (cs < W)
        if not in_bounds.any():
            continue
        rs = rs[in_bounds]
        cs = cs[in_bounds]
        # np.maximum.at is the right primitive: scattered max-stamp.
        np.maximum.at(height_map, (rs, cs), np.float32(h))
    return height_map


@dataclass
class Observation:
    """One (target, leaf-question-ids) pair the reasoner wants to resolve at a pose.

    Every waypoint carries a list of observations. Each observation
    names what the per-stop reasoner round should *look at* (a graph
    feature, a runtime detection, or a target pose for free-form
    discovery) and the ids of the leaf questions it bears on. A single
    pose can carry multiple observations when its FOV covers multiple
    targets.

    The mode (exploit / cover / discover) is a label the dashboard
    derives from the observation shape:

      - one observation, ``target_fid`` resolves to a high-priority
        candidate -> exploit
      - several observations spanning an anchor's neighbors -> cover
      - observations with ``target_fid`` and ``target_pose`` both
        ``None`` -> discover

    ``question_ids`` references leaf Question nodes that live on the
    contributing reasoner's Question forest. The runtime resolves
    the leaf payload when the per-stop reasoner round reads this stop's
    crops.
    """

    # The thing to observe. Exactly one of ``target_fid`` /
    # ``target_pose`` is typically set:
    #   - ``target_fid``: feature_id from the runtime graph
    #     (cityref_<N>, detected_<N>, road_<N>, building_<N>, ...).
    #   - ``target_pose``: world (x, y) when the observation is an
    #     area / region without a discrete graph feature (discovery
    #     mode, lawnmower coverage of an open space).
    # When both are ``None``, the observation is "look around at this
    # pose and answer the listed leaves about whatever is visible".
    target_fid: str | None = None
    target_pose: tuple[float, float] | None = None
    # Ids of leaf Question nodes the per-stop reasoner round bears on at
    # this observation. Resolved against the contributing reasoner's
    # Question forest at runtime.
    question_ids: list[str] = field(default_factory=list)
    # Reasoner's prior on importance of this observation, in [0, 1].
    # Aggregated across reasoners by the synthesizer.
    priority: float = 0.5
    # View hints: which face of the target the per-stop reasoner round
    # needs to see, which anchor the camera should look *from*, etc.
    # Same semantics as the previous waypoint-level fields but
    # per-target so the synthesizer's pose resolver can pick a pose
    # that satisfies all observations on a waypoint.
    view_intent: Literal["top_down", "side", "any"] = "any"
    view_face: ViewFace = "any"
    view_from_fid: str | None = None
    # Operator-facing label for the dashboard ("Building", "Willmore
    # Road", "light blue car"). Used for marker coloring and the
    # per-stop hover popup; not load-bearing for verification.
    intel_target: str = ""


@dataclass




@dataclass
class TourStop:
    """One stop on the synthesized tour.

    Like ``ProposedWaypoint``, a stop is a pose plus observations.
    ``aim_targets`` is the ordered list of distinct targets to sweep over
    at ``pose``. Each entry carries the target's world ``(x, y, z)`` and the
    observation priority that pulled it into this stop — a place, not a
    camera pose, because the crop is nadir. Empty
    when no observation in the stop has a resolvable ``target_fid``
    (cover / discover mode); the runtime holds nadir in that case.
    """

    stop_id: str
    pose: tuple[float, float, float, float]
    observe_radius_m: float
    observations: list[Observation] = field(default_factory=list)
    contributing_reasoners: list[dict] = field(default_factory=list)
    aim_targets: list[dict] = field(default_factory=list)
    # Authoritative per-leaf evidence payload the per-stop reasoner
    # round reads against. Each entry carries
    # ``{req_key, leaf_id, h_index, obs_idx, referent_fid, view_spec,
    #   question_text, attribute_axis, objects_to_mask, priority,
    #   intel_target, obs_*}``. Populated by the leaf-grained
    # collective synthesizer (and the per-obs fallback when collective
    # coverage is disabled).
    leaves: list[dict] = field(default_factory=list)
    # Per-backend picker diag (perspective: score breakdown; orthographic:
    # source / target_xy / altitude) so a post-mortem can read tour.json
    # and see why this pose won. Empty dict when no picker fired.
    pose_picker_diag: dict | None = None


def _snap_to_lattice(
    pose: tuple[float, float, float, float],
    *,
    forward_step_m: float = 5.0,
    yaw_step_deg: float = 30.0,
    alt_step_m: float = 2.0,
) -> tuple[float, float, float, float]:
    """Snap a continuous pose onto the CityNav agent's discrete lattice.

    The agent action space is ``MOVE_FORWARD 5 m``, ``TURN ±30°``,
    ``GO_UP / GO_DOWN ±2 m``. Snap to absolute world multiples of
    those steps so two stops with nearby continuous poses land on the
    same lattice node when they should and on different nodes when
    they shouldn't. Heading is wrapped into ``[0, 360)``.
    """
    x, y, alt_m, heading_deg = pose
    x_snap = round(float(x) / float(forward_step_m)) * float(forward_step_m)
    y_snap = round(float(y) / float(forward_step_m)) * float(forward_step_m)
    alt_snap = round(float(alt_m) / float(alt_step_m)) * float(alt_step_m)
    heading_snap = (
        round(float(heading_deg) / float(yaw_step_deg)) * float(yaw_step_deg)
    ) % 360.0
    return (x_snap, y_snap, alt_snap, heading_snap)


@dataclass
class _ResolverContext:
    """Inputs the per-observation pose resolver needs.

    Bundles ``runtime_graph``, ``bounds``, ``cell_size_m``,
    optional ``height_map`` and ``sensor_config`` plus the
    backend / altitude hint the viewpoint dispatcher needs so call
    sites pass one object instead of seven positional arguments.
    """

    runtime_graph: Any
    bounds: tuple[float, float, float, float] | None
    cell_size_m: float | None
    height_map: np.ndarray | None
    sensor_config: dict | str | None
    # Altitude hint the picker sizes the view square from
    # (``altitude_hint_m − ground_z``).
    start_altitude_m: float | None = None
    # Per-block upstream ground datum (``GROUND_LEVEL[map]``) the
    # orthographic picker uses to size the altitude-driven view square
    # (``side = 2 * (alt - ground)``). Supplied by the caller so the
    # picker never reads a per-object height prior. ``None`` for the
    # perspective backend, which ignores it.
    ortho_ground_z_m: float | None = None
    # Side-channel keyed by ``_observation_pose_key`` so the viewpoint
    # picker can surface its per-component score breakdown to the stop
    # record. The synthesiser reads this back and attaches it to the
    # TourStop after ``_resolve_observation_pose`` returns, so it lands
    # in tour.json for post-mortem.
    pose_picker_diags: dict[tuple, dict[str, Any]] = field(default_factory=dict)


def _observation_pose_key(obs: "Observation") -> tuple:
    """The 5-tuple the synthesiser uses to dedup observations.

    The synthesiser and the pose cache both key on this exact shape, so
    a pose computed during pre-warming hits the cache when the
    synthesiser later asks for it.
    """
    target_pose_key = (
        (float(obs.target_pose[0]), float(obs.target_pose[1]))
        if obs.target_pose is not None
        else None
    )
    return (
        obs.target_fid,
        target_pose_key,
        obs.view_intent,
        obs.view_face,
        obs.view_from_fid,
    )




def _node_centroid_rc(runtime_graph: Any, fid: str) -> tuple[int, int] | None:
    """Read ``(row, col)`` from the runtime graph's node entry."""
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(fid)
    if node is None:
        return None
    centroid = getattr(node, "centroid", None)
    if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
        return None
    try:
        return (int(round(float(centroid[0]))), int(round(float(centroid[1]))))
    except (TypeError, ValueError):
        return None


def _node_height_m(runtime_graph: Any, fid: str) -> float | None:
    """Read ``tags['height_m']`` (or ``height``) off the node, in meters."""
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(fid)
    if node is None:
        return None
    tags = getattr(node, "tags", None)
    if not isinstance(tags, dict):
        return None
    raw = tags.get("height_m")
    if raw is None:
        raw = tags.get("height")
    if raw is None:
        return None
    try:
        h = float(raw)
    except (TypeError, ValueError):
        return None
    return h if h > 0.0 else None


def _node_ground_z(runtime_graph: Any, fid: str) -> float:
    """World ground elevation at the node's footprint, in meters.

    The CityRefer bake stamps each feature with ``altitude_m`` (the
    world altitude of the object's TOP, per the sensaturban metadata)
    and ``height_m`` (object height); ``altitude_m - height_m`` is the
    ground elevation immediately under the footprint. Reading the
    height_map raster at the centroid is wrong here because that
    raster stamps the building's own height (so an inside-building
    centroid samples the roof, not the ground). Returns ``0.0`` when
    either tag is missing or unreadable.
    """
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return 0.0
    node = nodes.get(fid)
    if node is None:
        return 0.0
    tags = getattr(node, "tags", None)
    if not isinstance(tags, dict):
        return 0.0
    try:
        alt = float(tags.get("altitude_m", 0.0))
        h = float(tags.get("height_m", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return alt - h


def _node_principal_axis_deg(runtime_graph: Any, fid: str) -> float | None:
    """Read principal-axis orientation in degrees off the node."""
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(fid)
    if node is None:
        return None
    orientation = getattr(node, "orientation", None)
    if orientation is None:
        return None
    try:
        return float(orientation)
    except (TypeError, ValueError):
        return None


def _node_cells_rc(runtime_graph: Any, fid: str) -> np.ndarray | None:
    """Return the ``(N, 2) int64`` row/col footprint of a runtime-graph node.

    Reads ``node.cells`` (an ``(N, 2)`` row/col list that every prebake
    or detected node carries when its mask is realised on the tile
    grid). Returns ``None`` when the node is missing, has no cells, or
    the array can't be coerced to the expected shape.
    """
    nodes = getattr(runtime_graph, "nodes", None)
    if not isinstance(nodes, dict):
        return None
    node = nodes.get(fid)
    if node is None:
        return None
    cells = getattr(node, "cells", None)
    if cells is None:
        return None
    try:
        arr = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _cells_set(cells_arr: np.ndarray) -> set[tuple[int, int]]:
    """``(N, 2) -> set[(r, c)]`` for fast membership checks."""
    return {(int(r), int(c)) for r, c in cells_arr}


def _cardinal_facade_cells(cells_arr: np.ndarray, face: str) -> np.ndarray:
    """Return the subset of footprint cells lying on the named cardinal
    edge of the building.

    Convention matches ``grid_to_world_xy``: ``world_y = y_min +
    (row + 0.5) * cell_size_m``, so a LARGER row maps to a LARGER world
    y, i.e. more north. World x mirrors col exactly (LARGER col -> more
    east). The cardinal bearing table (``_VIEW_FACE_BEARING_DEG``)
    follows the standard compass: north = 0°, east = 90°, south = 180°,
    west = 270°. Picks:

      - north -> max row (largest world y on the footprint),
      - south -> min row,
      - east  -> max col,
      - west  -> min col.
    """
    rs = cells_arr[:, 0]
    cs = cells_arr[:, 1]
    if face == "north":
        return cells_arr[rs == rs.max()]
    if face == "south":
        return cells_arr[rs == rs.min()]
    if face == "east":
        return cells_arr[cs == cs.max()]
    if face == "west":
        return cells_arr[cs == cs.min()]
    return cells_arr


_FACADE_OUTWARD_NORMAL_RC = {
    # (dr, dc) one-cell step from the facade-midpoint cell into open
    # ground beyond the building. Matches the ``grid_to_world_xy``
    # convention (larger row -> larger world y -> more north;
    # larger col -> larger world x -> more east).
    "north": (1, 0),
    "south": (-1, 0),
    "east": (0, 1),
    "west": (0, -1),
}


def _count_exposed_rays(
    facade_cells: np.ndarray,
    outward_drc: tuple[int, int],
    probe_alt_m: float,
    height_map: np.ndarray,
    target_cells: set[tuple[int, int]],
    probe_steps: int = 8,
) -> int:
    """How many facade-midpoint rays of ``probe_steps`` cells outward are
    fully unblocked at altitude ``probe_alt_m``.

    Uses one sample at the geometric midpoint of the facade. Each step
    advances by ``outward_drc``; the test cell is blocked when
    ``height_map[r, c] > probe_alt_m`` and the cell is not in
    ``target_cells``. The probe is symmetric across the facade so the
    count peaks for facades that face open ground.
    """
    if facade_cells.size == 0:
        return 0
    H, W = int(height_map.shape[0]), int(height_map.shape[1])
    mid_r = int(round(float(facade_cells[:, 0].mean())))
    mid_c = int(round(float(facade_cells[:, 1].mean())))
    dr, dc = int(outward_drc[0]), int(outward_drc[1])
    free = 0
    for step in range(1, int(probe_steps) + 1):
        r = mid_r + dr * step
        c = mid_c + dc * step
        if not (0 <= r < H and 0 <= c < W):
            # Edge of the map counts as free space: nothing taller out
            # there to block the camera.
            free += 1
            continue
        if (r, c) in target_cells:
            continue
        if float(height_map[r, c]) > probe_alt_m:
            break
        free += 1
    return free


def _facade_midpoint_world(
    runtime_graph: Any,
    fid: str,
    view_face: ViewFace,
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
    height_map: np.ndarray,
    *,
    drone_xy_hint: tuple[float, float] | None = None,
) -> tuple[float, float, float, float, str] | None:
    """Return ``(fx, fy, fz, face_bearing_deg, face_name)`` for the
    facade the camera should aim at.

    For cardinal ``view_face`` (north / south / east / west), picks the
    edge cells on that side of the footprint and returns their world-XY
    midpoint at z = ground + 0.5 * h_m. For ``view_face == "principal"``,
    picks the two long-axis facades (perpendicular to the principal
    axis) and keeps the one with the most exposed-ray count. For
    ``view_face == "any"``, scores all four cardinals by exposed-ray
    count at the facade's own altitude and picks the highest (longest
    exposed wall heuristic).

    Returns ``None`` when the node has no recoverable footprint
    (e.g. discover-mode observations or detection-only nodes without
    cells). The face_bearing_deg follows the convention used elsewhere
    in this file: bearing from candidate centroid -> drone position
    (i.e. the *outward* direction from the candidate toward the
    facade-viewing drone).
    """
    cells_arr = _node_cells_rc(runtime_graph, fid)
    if cells_arr is None:
        return None
    H, W = int(height_map.shape[0]), int(height_map.shape[1])
    in_bounds = (
        (cells_arr[:, 0] >= 0)
        & (cells_arr[:, 0] < H)
        & (cells_arr[:, 1] >= 0)
        & (cells_arr[:, 1] < W)
    )
    if not in_bounds.any():
        return None
    cells_arr = cells_arr[in_bounds]
    target_cells = _cells_set(cells_arr)
    h_m = _node_height_m(runtime_graph, fid) or 0.0
    # ``height_map[centroid]`` stamps the building's own height for an
    # inside-building cell, so it points at the roof, not the ground.
    # Pull the world ground elevation from the node's ``altitude_m`` /
    # ``height_m`` tags so the facade z lands on mid-wall.
    ground_z_m = _node_ground_z(runtime_graph, fid)
    facade_z_m = ground_z_m + 0.5 * float(h_m)
    # Probe altitude: a viewer hovering at the building's roof plus a
    # margin. This is for ray-counting only; the actual drone altitude
    # is picked downstream.
    probe_alt_m = ground_z_m + float(h_m) + 5.0

    def _midpoint_world(face_cells: np.ndarray) -> tuple[float, float]:
        gr = float(face_cells[:, 0].mean())
        gc = float(face_cells[:, 1].mean())
        return grid_to_world_xy(gr, gc, bounds, float(cell_size_m))

    # Build the candidate-face list under each branch.
    if view_face in _VIEW_FACE_BEARING_DEG:
        face_name = str(view_face)
        face_cells = _cardinal_facade_cells(cells_arr, face_name)
        fx, fy = _midpoint_world(face_cells)
        bearing = float(_VIEW_FACE_BEARING_DEG[face_name])
        return (fx, fy, facade_z_m, bearing, face_name)

    if view_face == "principal":
        principal = _node_principal_axis_deg(runtime_graph, fid)
        if principal is not None:
            # Map principal axis onto its nearer cardinal pair; the
            # long-axis facades are perpendicular to the principal
            # axis, so the two viewing bearings are principal ± 90.
            principal_mod = float(principal) % 180.0
            if principal_mod <= 45.0 or principal_mod >= 135.0:
                cardinals = ("east", "west")
            else:
                cardinals = ("north", "south")
        else:
            cardinals = ("north", "south", "east", "west")
    else:
        cardinals = ("north", "south", "east", "west")

    # Score each candidate face by exposed-ray count and pick the best.
    # Approach-aligned tiebreak when a drone-position hint is given: of
    # the two highest-scoring faces, prefer the one whose outward normal
    # is closest to the drone's bearing back to the centroid.
    scored: list[tuple[int, str, np.ndarray]] = []
    for face_name in cardinals:
        face_cells = _cardinal_facade_cells(cells_arr, face_name)
        if face_cells.size == 0:
            continue
        free = _count_exposed_rays(
            face_cells,
            _FACADE_OUTWARD_NORMAL_RC[face_name],
            probe_alt_m,
            height_map,
            target_cells,
        )
        scored.append((free, face_name, face_cells))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_free, best_name, best_cells = scored[0]
    # If we have a drone hint and the top two are tied, prefer the
    # approach-aligned face. This keeps the aim offset small for the
    # common case where the reasoner authored view_face="any".
    if drone_xy_hint is not None and len(scored) >= 2 and scored[1][0] == best_free:
        gr_centroid = float(cells_arr[:, 0].mean())
        gc_centroid = float(cells_arr[:, 1].mean())
        cx, cy = grid_to_world_xy(gr_centroid, gc_centroid, bounds, float(cell_size_m))
        approach_bearing = (
            math.degrees(
                math.atan2(float(drone_xy_hint[0]) - cx, float(drone_xy_hint[1]) - cy)
            )
            + 360.0
        ) % 360.0
        best_delta = 999.0
        for _free, name, cells_subset in scored:
            if _free != best_free:
                break
            bearing = float(_VIEW_FACE_BEARING_DEG[name])
            delta = abs(((approach_bearing - bearing + 180.0) % 360.0) - 180.0)
            if delta < best_delta:
                best_delta = delta
                best_name = name
                best_cells = cells_subset
    fx, fy = _midpoint_world(best_cells)
    bearing = float(_VIEW_FACE_BEARING_DEG[best_name])
    return (fx, fy, facade_z_m, bearing, best_name)


def _resolve_aim_targets(
    stop: "TourStop",
    runtime_graph: Any,
    height_map: np.ndarray | None,
    bounds: tuple[float, float, float, float] | None,
    cell_size_m: float | None,
) -> list[dict]:
    """Resolve every distinct target_fid in the stop into a world aim target.

    Walks the stop's observations in priority order; for each new fid,
    builds the aim target's ``(x, y, z)``:

      - ``view_intent == "side"``: aim at the chosen facade midpoint
        via :func:`_facade_midpoint_world` so the aim lands on facade
        pixels rather than the building's roof. Falls back to
        the centroid aim when the footprint cells aren't available.
      - everything else: aim at the centroid at
        ``z = ground_z + h_m / 2``.

    Returns ``{fid, x, y, z, priority}`` per distinct aim target, ordered
    by descending observation priority. Empty list when no observation in
    the stop has a resolvable target_fid (cover / discover mode).
    """
    if (
        runtime_graph is None
        or height_map is None
        or bounds is None
        or cell_size_m is None
    ):
        return []
    H, W = int(height_map.shape[0]), int(height_map.shape[1])
    out: list[dict] = []
    seen_fids: set[str] = set()
    for obs in stop.observations:
        fid = obs.target_fid
        if not fid or fid in seen_fids:
            continue
        aim: tuple[float, float, float] | None = None
        if obs.view_intent == "side":
            facade = _facade_midpoint_world(
                runtime_graph,
                fid,
                obs.view_face,
                bounds,
                float(cell_size_m),
                height_map,
                drone_xy_hint=(float(stop.pose[0]), float(stop.pose[1])),
            )
            if facade is not None:
                fx, fy, fz, _bearing_deg, _face_name = facade
                aim = (float(fx), float(fy), float(fz))
        if aim is None:
            centroid_rc = _node_centroid_rc(runtime_graph, fid)
            if centroid_rc is None:
                continue
            row, col = centroid_rc
            if not (0 <= row < H and 0 <= col < W):
                continue
            x, y = grid_to_world_xy(float(row), float(col), bounds, float(cell_size_m))
            ground_z = _node_ground_z(runtime_graph, fid)
            h_m = _node_height_m(runtime_graph, fid) or 0.0
            aim = (float(x), float(y), ground_z + 0.5 * float(h_m))
        ax, ay, az = aim
        out.append(
            {
                "fid": str(fid),
                "x": ax,
                "y": ay,
                "z": az,
                "priority": float(obs.priority),
            }
        )
        seen_fids.add(fid)
    return out


def _attach_aim_targets(
    stops: list["TourStop"],
    *,
    runtime_graph: Any,
    height_map: np.ndarray | None,
    bounds: tuple[float, float, float, float] | None,
    cell_size_m: float | None,
) -> None:
    """Bake ``aim_targets`` (one per resolved target) onto each stop, in place.

    Each distinct ``target_fid`` in the stop's observations becomes its
    own aim target; the runtime sweep visits them in priority order and
    uses ``SEARCH_ANCHOR_DWELL_FRAMES`` (read at sweep time, not baked) to
    gate per-target settlement. Stops with no resolvable target land on an
    empty ``aim_targets`` list.

    The ortho crop is altitude-driven orthographic — pitch is always nadir —
    so an aim target is a place and carries no camera pose.
    """
    for stop in stops:
        aim_targets = _resolve_aim_targets(
            stop, runtime_graph, height_map, bounds, cell_size_m
        )
        if not aim_targets:
            stop.aim_targets = []
            continue
        stop.aim_targets = [
            {
                "fid": str(aim["fid"]),
                "x": float(aim["x"]),
                "y": float(aim["y"]),
                "z": float(aim["z"]),
                "priority": float(aim.get("priority", 0.0)),
            }
            for aim in aim_targets
        ]


def _runtime_graph_to_viewpoint_features(
    runtime_graph: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Adapt a runtime ObjectGraph into the per-class feature-dict shape
    the viewpoint pickers consume (``perception.viewpoint``).

    Each ObjectGraph node (a ``MapObject``) carries the fields we need:
    ``cells`` (sparse (row, col) array), ``centroid``, ``class_name``,
    ``name``, and ``tags`` (with ``altitude_m`` / ``height_m`` as strings
    from the bake). CityRefer nodes keep their numeric ``cityref_id``;
    OSM/runtime nodes carry their native ``fid`` so the same viewpoint
    path can target ``road_*`` / ``building_*`` / ``detected_*`` ids.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    nodes = getattr(runtime_graph, "nodes", None) or {}
    n_total = 0
    n_bad_cityref_id = 0
    n_no_cells = 0
    n_empty_cells = 0
    n_bad_cells_shape = 0
    for nid, node in nodes.items():
        n_total += 1
        fid = str(nid)
        cid: int | str | None = None
        is_cityref = fid.startswith("cityref_")
        if is_cityref:
            suffix = fid.split("_", 1)[1]
            if not suffix:
                n_bad_cityref_id += 1
                continue
            try:
                cid = int(suffix)
            except ValueError:
                cid = suffix
        cells = getattr(node, "cells", None)
        # Detected nodes carry no `cells`, and priors may have no
        # mask-cache row. The ortho backend grounds a viewpoint's target_xy from the node
        # centroid alone, so a cells-less node must NOT be dropped here — emit a
        # centroid-only feature.
        node_centroid = getattr(node, "centroid", None)
        centroid_only = cells is None
        arr = None
        if centroid_only:
            if not node_centroid or len(node_centroid) < 2:
                n_no_cells += 1
                continue
            centroid = (int(node_centroid[0]), int(node_centroid[1]))
        else:
            try:
                arr = np.asarray(cells, dtype=np.int32).reshape(-1, 2)
            except Exception:
                n_bad_cells_shape += 1
                continue
            if arr.size == 0:
                n_empty_cells += 1
                continue
            centroid = node_centroid or (
                int(round(arr[:, 0].mean())),
                int(round(arr[:, 1].mean())),
            )
        tags = getattr(node, "tags", {}) or {}
        try:
            altitude_m = float(tags.get("altitude_m", 0.0))
        except (TypeError, ValueError):
            altitude_m = 0.0
        try:
            height_m = float(tags.get("height_m", 0.0))
        except (TypeError, ValueError):
            height_m = 0.0
        class_name = str(getattr(node, "class_name", "") or "unknown")
        feat = {
            "name": getattr(node, "name", None) or "",
            "fid": fid,
            "cells": [] if centroid_only else arr.tolist(),
            "centroid_row": int(centroid[0]),
            "centroid_col": int(centroid[1]),
            "altitude_m": altitude_m,
            "height_m": height_m,
        }
        if cid is not None:
            feat["cityref_id"] = cid
        # Indoor room link: the ground-level FPV picker needs each
        # object's parent room (parent_fid) and the object_type / room_type
        # tags to resolve the room's free floor. They live on the baked
        # node tags; forward them when present so the runtime/flight path
        # feeds the picker the same signal the scenario-editor path reads
        # straight from _features.json. Aerial pickers ignore these keys.
        for tag_key in ("parent_fid", "object_type", "room_type"):
            tag_val = tags.get(tag_key)
            if tag_val is not None:
                feat[tag_key] = tag_val
        out.setdefault(class_name, []).append(feat)
    if not out:
        log.warning(
            "_runtime_graph_to_viewpoint_features: empty result; nodes_seen=%d "
            "bad_cityref_id=%d no_cells=%d empty_cells=%d "
            "bad_cells_shape=%d",
            n_total,
            n_bad_cityref_id,
            n_no_cells,
            n_empty_cells,
            n_bad_cells_shape,
        )
    return out


def _target_xy_from_features(
    features: Mapping[str, Sequence[Mapping[str, Any]]],
    target_fid: str,
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
) -> tuple[float, float] | None:
    """Resolve a target fid's world (x, y) centroid from the feature dict.

    The orthographic picker uses this centroid to pin the drone's nadir
    view directly over the target.
    """
    bx0, by0 = float(bounds[0]), float(bounds[1])
    cs = float(cell_size_m)
    for _cls, feats in features.items():
        for ft in feats:
            fid = ft.get("fid")
            if fid is not None and str(fid) != target_fid:
                continue
            cid = ft.get("cityref_id")
            if fid is None and (cid is None or f"cityref_{cid}" != target_fid):
                continue
            cr = ft.get("centroid_row")
            cc = ft.get("centroid_col")
            if cr is None or cc is None:
                return None
            return (
                bx0 + (float(cc) + 0.5) * cs,
                by0 + (float(cr) + 0.5) * cs,
            )
    return None


def _build_viewpoint_request(
    obs: Observation,
    ctx: _ResolverContext,
    cfg: TourSynthesizerConfig,
) -> Any | None:
    """Build a ``perception.viewpoint.ViewpointRequest`` for one observation.

    Returns ``None`` when the observation can't be turned into a request
    (no target_fid, missing context, perception import failure, features
    empty, centroid missing).
    """
    if obs.target_fid is None:
        log.warning(
            "_build_viewpoint_request: obs.target_fid is None "
            "(view_from=%s view_intent=%s view_face=%s); returning None",
            obs.view_from_fid,
            obs.view_intent,
            obs.view_face,
        )
        return None
    if ctx.bounds is None or ctx.cell_size_m is None:
        log.warning(
            "_build_viewpoint_request: target=%s missing ctx "
            "(bounds=%r cell_size=%r); returning None",
            obs.target_fid,
            ctx.bounds,
            ctx.cell_size_m,
        )
        return None
    try:
        from maps.perception.viewpoint import ViewpointRequest
    except Exception:  # noqa: BLE001
        log.warning(
            "_build_viewpoint_request: target=%s perception import failed; "
            "returning None",
            obs.target_fid,
            exc_info=True,
        )
        return None

    # Grounding-free view-position path: an explicit ``target_pose`` (set for
    # a view-position sentinel, resolved by the compiler from the drone/start
    # pose or the operator marker) drives ``target_xy`` DIRECTLY -- the target
    # has no grounded node, so skip the graph-features centroid resolution.
    # The orthographic picker needs only ``target_xy``.
    if obs.target_pose is not None:
        target_xy = (float(obs.target_pose[0]), float(obs.target_pose[1]))
    else:
        features = _runtime_graph_to_viewpoint_features(ctx.runtime_graph)
        if not features:
            log.warning(
                "_build_viewpoint_request: target=%s features dict empty after "
                "_runtime_graph_to_viewpoint_features (see prior warning for counts); "
                "returning None",
                obs.target_fid,
            )
            return None

        target_xy = _target_xy_from_features(
            features, obs.target_fid, ctx.bounds, ctx.cell_size_m
        )
        if target_xy is None:
            log.warning(
                "_build_viewpoint_request: target=%s no centroid in features; returning None",
                obs.target_fid,
            )
            return None

    altitude_hint_m = (
        float(ctx.start_altitude_m)
        if ctx.start_altitude_m is not None
        else float(cfg.default_altitude_hint_m)
    )

    return ViewpointRequest(
        target_fid=obs.target_fid,
        target_xy=target_xy,
        altitude_hint_m=altitude_hint_m,
    )


def _pick_observation_pose(
    obs: Observation,
    ctx: _ResolverContext,
    cfg: TourSynthesizerConfig,
) -> tuple[float, float, float, float] | None:
    """Per-observation pose picker, over ``perception.viewpoint``.

    Returns the canonical ``(x, y, altitude_m, heading_deg)`` tuple
    ``_snap_to_lattice`` consumes, or ``None`` when the picker finds
    no clear pose (caller drops the observation).
    """
    req = _build_viewpoint_request(obs, ctx, cfg)
    if req is None:
        return None
    try:
        from maps.perception.viewpoint import pick_viewpoint
    except Exception:  # noqa: BLE001
        log.warning(
            "_pick_observation_pose: target=%s perception import failed; "
            "returning None",
            obs.target_fid,
            exc_info=True,
        )
        return None

    result = pick_viewpoint(req)
    if result is None:
        log.warning(
            "_pick_observation_pose: target=%s view_from=%s "
            "pick_viewpoint returned None; dropping observation",
            obs.target_fid,
            obs.view_from_fid,
        )
        return None

    log.info(
        "viewpoint pose target=%s view_from=%s "
        "pose=(%.1f, %.1f, %.1f) yaw=%.1f source=%s",
        obs.target_fid,
        obs.view_from_fid,
        result.pose_xy[0],
        result.pose_xy[1],
        result.altitude_m,
        result.yaw_deg,
        result.diag.get("source"),
    )

    # Stash the JSON-serialisable diag in the resolver-context side-
    # channel keyed by the observation's dedup tuple. The synthesiser
    # reads this back after _resolve_observation_pose returns and
    # attaches it to the TourStop. The picker already coerced numpy
    # scalars to Python types before returning.
    ctx.pose_picker_diags[_observation_pose_key(obs)] = dict(result.diag)
    return (
        float(result.pose_xy[0]),
        float(result.pose_xy[1]),
        float(result.altitude_m),
        float(result.yaw_deg),
    )


def _resolve_observation_pose(
    obs: Observation,
    ctx: _ResolverContext,
    cfg: TourSynthesizerConfig,
) -> tuple[float, float, float, float] | None:
    """Pick a lattice-snapped drone pose for one observation.

    Returns ``None`` when the observation can't be resolved (its
    ``target_fid`` has no node in the runtime graph and there is no
    ``target_pose``). The caller drops unresolved observations.
    """
    # The voxel-grid pose picker is the only path. It enforces the
    # airspace check (drone xy not inside any feature's footprint with
    # cell_top within ``headroom_m`` of drone altitude), the "drone on
    # a line from any X-cell to any Y-cell, never over X" geometric
    # constraint when ``view_from_fid`` is set, the Johnson +
    # Massios-Fisher + Koopman + composition score with per-task face
    # weights, and a local-search refinement around the winner.
    # Returns None when the picker can't find any clear pose (target
    # absent from the voxel grid, every candidate failed the airspace
    # check, no rays hit the target). The caller drops the observation
    # in that case -- matches the pre-existing "unresolvable
    # observation" contract.
    picked_pose = _pick_observation_pose(obs, ctx, cfg)
    if picked_pose is None:
        log.warning(
            "_resolve_observation_pose: viewpoint picker returned None "
            "for target=%s view_from=%s view_intent=%s view_face=%s "
            "(see prior _pick_observation_pose warnings); dropping observation",
            obs.target_fid,
            obs.view_from_fid,
            obs.view_intent,
            obs.view_face,
        )
        return None
    return _snap_to_lattice(
        picked_pose,
        forward_step_m=cfg.agent_forward_step_m,
        yaw_step_deg=cfg.agent_yaw_step_deg,
        alt_step_m=cfg.agent_altitude_step_m,
    )


































def _safe_dedup_exempt(pred, target_fid, pose_xy) -> bool:
    """Probe the boost dedup-bypass predicate defensively: a raising
    predicate is treated as NON-exempt (normal dedup applies) and logged —
    a best-effort bypass must never take the whole extension down with it
    (its failure would otherwise be swallowed as a benign synth error and
    silently drop EVERY requested viewpoint in the batch)."""
    if not callable(pred):
        return False
    try:
        return bool(pred(target_fid, pose_xy))
    except Exception:  # noqa: BLE001 — bypass is best-effort, never fatal
        log.exception(
            "dedup_exempt predicate raised for %s; treating as non-exempt",
            target_fid,
        )
        return False


def synthesize_extension(
    *,
    existing_tour: list[TourStop],
    viewpoints_to_add: list[dict],
    runtime_graph: Any,
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
    cfg: TourSynthesizerConfig,
    height_map: np.ndarray | None = None,
    sensor_config: dict | str | None = None,
    start_altitude_m: float | None = None,
    ortho_ground_z_m: float | None = None,
    dedup_exempt: Any = None,
) -> list[TourStop]:
    """Append free-form viewpoints to a fully-walked tour.

    Given free-form ``(target_fid, view_intent, view_face)`` tuples,
    this projects each tuple through the same
    :func:`_resolve_observation_pose` resolver Stage 2 uses, dedups
    against the tour's existing stops on the request key and on
    same-target resolved pose, and appends new stops in request order
    (no re-routing of the walked prefix).

    ``dedup_exempt`` (dedup BYPASS): an optional predicate
    ``(target_fid, pose_xy | None) -> bool``. A re-request whose intent
    key / resolved pose would dedup as already-flown is STILL scheduled
    when the predicate returns True — the contract is that a pending
    BOOST on that stop/area makes the re-flight a new perception act
    (new vocabulary), so absorbing it would silently drop the agent's
    perception retarget. No caller supplies one today -- the round-era tour
    extension was its only source -- so the parameter is the seam a future
    re-flight policy plugs into, not a live behavior.

    Returns ``existing_tour + new_stops``. When every requested
    viewpoint dedups against an existing stop or cannot be resolved,
    the return value is ``existing_tour`` unchanged.

    ``viewpoints_to_add`` entries are dicts with at least
    ``target_fid``; ``view_intent`` defaults to ``"any"`` and
    ``view_face`` to ``"any"`` when absent.

    (Legacy mid-flight tour-extension path; the cross-reasoner tour-end
    judge that drove it has been removed.)
    """
    if not viewpoints_to_add:
        return list(existing_tour)

    ctx = _ResolverContext(
        runtime_graph=runtime_graph,
        bounds=bounds,
        cell_size_m=cell_size_m,
        height_map=height_map,
        sensor_config=sensor_config,
        start_altitude_m=start_altitude_m,
        ortho_ground_z_m=ortho_ground_z_m,
    )

    existing_keys: set[tuple] = set()
    existing_pose_target_keys: set[tuple[str, tuple[float, float, float, float]]] = set()
    for stop in existing_tour:
        pose_key = cast(
            tuple[float, float, float, float],
            tuple(round(float(v), 3) for v in stop.pose),
        )
        for obs in stop.observations:
            existing_keys.add(
                (
                    obs.target_fid,
                    obs.view_intent,
                    obs.view_face,
                    obs.view_from_fid,
                )
            )
            if obs.target_fid:
                existing_pose_target_keys.add((obs.target_fid, pose_key))

    # Re-look exemption: when the batch carries EXACTLY ONE request and the
    # runtime graph holds ZERO detections, never dedupe it against an
    # already-flown key/pose. A blind flight's only recovery is a fresh
    # perception pass, and SAM zero-det at altitude is measured stochastic
    # (0 vs 210 dets at the same pose), so absorbing the sole re-look forces
    # a blind commit. Dedupe relaxation ONLY — no altitude/standoff
    # parameter (the ortho sensor already renders max sim resolution).
    _detected_nodes = 0
    _nodes = getattr(runtime_graph, "nodes", None)
    if _nodes is not None and hasattr(_nodes, "values"):
        _detected_nodes = sum(
            1 for _n in _nodes.values()
            if getattr(_n, "provenance", "") == "detected"
        )
    _well_formed = [
        v for v in viewpoints_to_add
        if isinstance(v, dict)
        and isinstance(v.get("target_fid"), str) and v["target_fid"].strip()
    ]
    relook_exempt = len(_well_formed) == 1 and _detected_nodes == 0
    if relook_exempt:
        log.info(
            "synthesize_extension: re-look exemption ACTIVE (sole pending "
            "request, detected_nodes=0) -- already-flown dedupe bypassed for "
            "target_fid=%s",
            str(_well_formed[0].get("target_fid") or "").strip(),
        )

    new_stops: list[TourStop] = []
    new_stops_by_pose: dict[tuple[float, float, float, float], TourStop] = {}
    next_idx = len(existing_tour)
    seen_in_extension: set[tuple] = set()
    for vp in viewpoints_to_add:
        if not isinstance(vp, dict):
            continue
        target_fid = vp.get("target_fid")
        if not isinstance(target_fid, str) or not target_fid.strip():
            continue
        target_fid = target_fid.strip()
        view_intent = str(vp.get("view_intent") or "any")
        if view_intent not in ("top_down", "side", "any"):
            view_intent = "any"
        view_face = str(vp.get("view_face") or "any")
        if view_face not in ("north", "south", "east", "west", "principal", "any"):
            view_face = "any"
        view_from_fid_raw = vp.get("view_from_fid")
        view_from_fid = (
            view_from_fid_raw.strip()
            if isinstance(view_from_fid_raw, str) and view_from_fid_raw.strip()
            else None
        )
        key = (target_fid, view_intent, view_face, view_from_fid)
        # Grounding-free view-position viewpoint: the compiler injects an
        # explicit ``target_pose`` [x, y] for a sentinel target_fid
        # (current_position / marked_region) that has no grounded prior node.
        # When present it drives the pose directly (see
        # ``_build_viewpoint_request``), so the item resolves to a nadir ortho
        # stop without a fid on the graph.
        vp_pose = vp.get("target_pose")
        if vp_pose is None:
            # UNIFORM ADDRESSING: an "[x, y]"-literal target_fid IS the pose
            # target — the agent points at a world coordinate (a derived
            # crossing, a terminus) that has no grounded node. Same
            # target_pose path the view-position sentinels ride.
            _pxy = parse_xy_token(target_fid)
            if _pxy is not None:
                vp_pose = [_pxy[0], _pxy[1]]
        obs_target_pose: tuple[float, float] | None = None
        if isinstance(vp_pose, (list, tuple)) and len(vp_pose) == 2:
            try:
                obs_target_pose = (float(vp_pose[0]), float(vp_pose[1]))
            except (TypeError, ValueError):
                obs_target_pose = None
        obs = Observation(
            target_fid=target_fid,
            target_pose=obs_target_pose,
            question_ids=[],
            priority=0.5,
            view_intent=cast(ViewIntent, view_intent),
            view_face=cast(ViewFace, view_face),
            view_from_fid=view_from_fid,
            intel_target="judge_extension",
        )
        pose = None
        if (key in existing_keys or key in seen_in_extension) and not relook_exempt:
            # Boost bypass: a pending boost on this target/area makes the
            # EXPLICIT re-request a NEW perception act — schedule it anyway.
            # Probe fid-first, then with the RESOLVED pose so an [x, y]-
            # targeted boost also matches; a raising predicate is treated
            # as non-exempt, never allowed to take the whole extension down.
            exempt = _safe_dedup_exempt(dedup_exempt, target_fid, None)
            if not exempt:
                pose = _resolve_observation_pose(obs, ctx, cfg)
                if pose is not None:
                    exempt = _safe_dedup_exempt(
                        dedup_exempt, target_fid,
                        (float(pose[0]), float(pose[1])))
            if not exempt:
                continue
            log.info(
                "synthesize_extension: dedup BYPASS (pending boost) for "
                "re-requested target_fid=%s (intent key already flown)",
                target_fid,
            )
        if pose is None:
            pose = _resolve_observation_pose(obs, ctx, cfg)
        if pose is None:
            if str(target_fid).startswith("detected_"):
                # The orthographic picker grounds a detected-fid viewpoint from
                # the node centroid, so a pose normally resolves; this WARN
                # makes a rare unresolvable pose observable instead of silent.
                log.warning(
                    "synthesize_extension: dropped detected-fid viewpoint "
                    "target_fid=%s (no resolvable pose)",
                    target_fid,
                )
            continue
        pose_key = cast(
            tuple[float, float, float, float],
            tuple(round(float(v), 3) for v in pose),
        )
        if (target_fid, pose_key) in existing_pose_target_keys and not relook_exempt:
            # Boost bypass, pose arm: an already-flown resolved pose with
            # a pending boost for that pose/area re-flies on the EXPLICIT
            # re-request (vocabulary alone never re-flies).
            if not _safe_dedup_exempt(
                    dedup_exempt, target_fid, (pose_key[0], pose_key[1])):
                continue
            log.info(
                "synthesize_extension: dedup BYPASS (pending boost) for "
                "re-requested pose %s of target_fid=%s",
                (pose_key[0], pose_key[1]), target_fid,
            )
        seen_in_extension.add(key)
        existing_new_stop = new_stops_by_pose.get(pose_key)
        if existing_new_stop is not None:
            existing_new_stop.observations.append(obs)
            continue
        stop = TourStop(
            stop_id=f"stop_{next_idx:03d}",
            pose=pose,
            observe_radius_m=cfg.fov_m / 2.0,
            observations=[obs],
            contributing_reasoners=[],
            aim_targets=[],
            leaves=[],
        )
        new_stops.append(stop)
        new_stops_by_pose[pose_key] = stop
        next_idx += 1

    if not new_stops:
        return list(existing_tour)

    _attach_aim_targets(
        new_stops,
        runtime_graph=runtime_graph,
        height_map=height_map,
        bounds=bounds,
        cell_size_m=cell_size_m,
    )
    return list(existing_tour) + new_stops


def build_initial_tour_from_viewpoints(
    *,
    requested_viewpoints: list[dict],
    runtime_graph: Any,
    start_xy: tuple[float, float],
    bounds: tuple[float, float, float, float],
    cell_size_m: float,
    cfg: TourSynthesizerConfig,
    height_map: np.ndarray | None = None,
    sensor_config: dict | str | None = None,
    start_altitude_m: float | None = None,
    ortho_ground_z_m: float | None = None,
) -> list[TourStop]:
    """Build the initial tour from round 0's requested viewpoints.

    The reasoner grounds the operator's named anchors over the PRIOR
    graph, then its first act files ``requested_viewpoints`` directly —
    the SAME item shape the in-flight tour extension consumes
    (``{target_fid, view_intent?, view_face?, view_from_fid?, rationale?}``).
    This runs them through :func:`synthesize_extension` from an EMPTY tour,
    so the initial tour and every in-flight extension are built by ONE
    machinery (same pose resolver, same aim attach).

    Returns ``list[TourStop]`` — the shape the compiler consumes — or
    ``[]`` when no requested viewpoint resolves to a pose.

    ``ortho`` (orthographic, nadir crop) keeps request order, matching
    upstream CityNav (no view-pose optimization).
    """
    if not requested_viewpoints:
        return []
    stops = synthesize_extension(
        existing_tour=[],
        viewpoints_to_add=requested_viewpoints,
        runtime_graph=runtime_graph,
        bounds=bounds,
        cell_size_m=cell_size_m,
        cfg=cfg,
        height_map=height_map,
        sensor_config=sensor_config,
        start_altitude_m=start_altitude_m,
        ortho_ground_z_m=ortho_ground_z_m,
    )
    if not stops:
        return []
    return stops


def _observation_to_dict(obs: Observation) -> dict:
    return {
        "target_fid": obs.target_fid,
        "target_pose": (
            [float(obs.target_pose[0]), float(obs.target_pose[1])]
            if obs.target_pose is not None
            else None
        ),
        "question_ids": [str(qid) for qid in obs.question_ids if isinstance(qid, str)],
        "priority": float(obs.priority),
        "view_intent": obs.view_intent,
        "view_face": obs.view_face,
        "view_from_fid": obs.view_from_fid,
        "intel_target": obs.intel_target,
    }


def _dict_to_observation(payload: dict) -> Observation:
    tp = payload.get("target_pose")
    target_pose: tuple[float, float] | None = None
    if isinstance(tp, (list, tuple)) and len(tp) == 2:
        try:
            target_pose = (float(tp[0]), float(tp[1]))
        except (TypeError, ValueError):
            target_pose = None
    raw_ids = payload.get("question_ids") or []
    question_ids: list[str] = []
    if isinstance(raw_ids, list):
        seen: set[str] = set()
        for qid in raw_ids:
            if not isinstance(qid, str):
                continue
            qid = qid.strip()
            if not qid or qid in seen:
                continue
            seen.add(qid)
            question_ids.append(qid)
    return Observation(
        target_fid=payload.get("target_fid"),
        target_pose=target_pose,
        question_ids=question_ids,
        priority=float(payload.get("priority") or 0.0),
        view_intent=cast(ViewIntent, str(payload.get("view_intent", "any"))),
        view_face=cast(ViewFace, str(payload.get("view_face", "any"))),
        view_from_fid=payload.get("view_from_fid"),
        intel_target=str(payload.get("intel_target", "")),
    )


def _aim_target_to_dict(aim: dict) -> dict:
    """JSON-safe shape for one ``TourStop`` aim-target entry."""
    return {
        "fid": str(aim["fid"]),
        "x": float(aim["x"]),
        "y": float(aim["y"]),
        "z": float(aim["z"]),
        "priority": float(aim.get("priority", 0.0)),
    }


def _dict_to_aim_target(payload: dict) -> dict:
    """Inverse of :func:`_aim_target_to_dict`. Skips entries missing a fid."""
    if not isinstance(payload, dict):
        return {}
    fid = payload.get("fid")
    if not isinstance(fid, str):
        return {}
    return {
        "fid": str(fid),
        "x": float(payload.get("x", 0.0)),
        "y": float(payload.get("y", 0.0)),
        "z": float(payload.get("z", 0.0)),
        "priority": float(payload.get("priority", 0.0)),
    }


def tour_stop_to_dict(stop: TourStop) -> dict:
    x, y, alt_m, heading = stop.pose
    out: dict = {
        "stop_id": stop.stop_id,
        "pose": {
            "x": float(x),
            "y": float(y),
            "alt_m": float(alt_m),
            "heading_deg": float(heading),
        },
        "observe_radius_m": float(stop.observe_radius_m),
        "observations": [_observation_to_dict(o) for o in stop.observations],
        "contributing_reasoners": list(stop.contributing_reasoners),
        "aim_targets": [_aim_target_to_dict(a) for a in stop.aim_targets],
        "leaves": [dict(leaf) for leaf in stop.leaves],
    }
    if stop.pose_picker_diag is not None:
        out["pose_picker_diag"] = dict(stop.pose_picker_diag)
    return out


def tour_stop_from_dict(payload: dict) -> TourStop:
    pose = payload["pose"]
    if "aim_targets" not in payload and "anchors" in payload:
        raise RuntimeError(
            "pre-rename tour.json artifact: this stop carries the legacy "
            "'anchors' wire key; the runtime reads 'aim_targets'. Re-bake "
            "the tour with current code (a frozen tour is only resumable "
            "on the code that baked it)."
        )
    aim_payload = payload.get("aim_targets") or []
    aim_targets: list[dict] = []
    if isinstance(aim_payload, list):
        for entry in aim_payload:
            cleaned = _dict_to_aim_target(entry if isinstance(entry, dict) else {})
            if cleaned:
                aim_targets.append(cleaned)
    leaves_payload = payload.get("leaves") or []
    leaves: list[dict] = [
        dict(leaf) for leaf in leaves_payload if isinstance(leaf, dict)
    ]
    diag_payload = payload.get("pose_picker_diag")
    return TourStop(
        stop_id=str(payload["stop_id"]),
        pose=(
            float(pose["x"]),
            float(pose["y"]),
            float(pose["alt_m"]),
            float(pose["heading_deg"]),
        ),
        observe_radius_m=float(payload.get("observe_radius_m", 15.0)),
        observations=[
            _dict_to_observation(o) for o in payload.get("observations") or []
        ],
        contributing_reasoners=list(payload.get("contributing_reasoners") or []),
        aim_targets=aim_targets,
        leaves=leaves,
        pose_picker_diag=dict(diag_payload) if isinstance(diag_payload, dict) else None,
    )






def write_tour_json(stops: list[TourStop], path: Path, header: dict) -> None:
    """Atomically write the per-turn tour artifact.

    ``header`` carries the schema-level metadata (turn, operator_text,
    intel_text, utility_text, spatial_scope, altitude_m, ordering_policy,
    fallback_mode). The function fills in ``schema_version`` and
    ``synthesizer_version`` for the caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": TOUR_SCHEMA_VERSION,
        "synthesizer_version": TOUR_SYNTHESIZER_VERSION,
        **header,
        "stops": [tour_stop_to_dict(s) for s in stops],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)




__all__ = [
    "TOUR_SCHEMA_VERSION",
    "TOUR_SYNTHESIZER_VERSION",
    "Observation",
    "TourStop",
    "TourSynthesizerConfig",
    "build_initial_tour_from_viewpoints",
    "synthesize_extension",
    "tour_stop_from_dict",
    "tour_stop_to_dict",
    "write_tour_json",
]
