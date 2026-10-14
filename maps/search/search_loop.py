"""Per-episode search state: the one world graph a mission reads and folds
each flight's detections into, and the segmenter's class set."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from maps.search.parse_compile.perception_director import (
    PerceptionDirector, canon_concept)
from maps.sws.graph.model import GridDomain, MapObject
from maps.util.mask_codec import encode_mask_record


log = logging.getLogger(__name__)

_NULL_CONCEPTS = frozenset({"none", "null", "n/a", "na"})


@dataclass
class _State:
    """Mutable internal state, read and written under ``SearchLoop._lock``."""

    runtime_graph: object  # ObjectGraph
    # {fid, class, xy, conf} the last update_perception folded: the flight
    # report serves them.
    last_flight_detections: list[dict] = field(default_factory=list)
    # The PerceptionDirector: the watchlist and the boost.
    perception_director: Any = None


class SearchLoop:
    """One episode's world graph, its tile frame and the segmenter hooks."""

    def __init__(
        self,
        grid: GridDomain,
        *,
        tile_origin_xy: tuple[float, float],
        map_name: str,
        sam_setter: Callable[[list[str]], None] | None,
        ortho_ground_z_m: float,
        rgbd_dir: str,
    ):
        self._grid = grid
        self._tile_origin_xy = (
            float(tile_origin_xy[0]),
            float(tile_origin_xy[1]),
        )
        self._map_name = map_name
        # The segmenter's concept list: the declared union and the director's
        # watchlist both write it whole.
        self._sam_setter = sam_setter
        # The block's ground datum: the observation span is measured off it.
        self._ortho_ground_z_m = ortho_ground_z_m
        # The raster ``look`` crops on the bench.
        self._rgbd_dir = rgbd_dir
        #: What a live run has captured (``benchmarks/flight/run_imagery``),
        #: cropped by ``look`` instead of the raster; None on the bench.
        self.look_imagery = None
        from maps.perception.world_tracker import WorldTracker

        self._world_tracker = WorldTracker()
        self._state: _State | None = None
        # Each prior's served row point, built on first use.
        self._prior_xy_cache: dict | None = None
        # Every public reader and writer of ``_State`` takes this lock;
        # re-entrant so a locked path may call another public method.
        self._lock = threading.RLock()

    def start(self, graph) -> None:
        """Bind the episode's graph (the prior seed every detection folds
        into) and an empty watchlist."""
        self._state = _State(runtime_graph=graph,
                             perception_director=PerceptionDirector(push=self._sam_setter))

    @property
    def _cell_size_m(self) -> float:
        """The tile grid's cell size in metres."""
        return float(self._grid.cell_size_m)

    def _grid_centroid_to_world_xy(
        self, centroid: object
    ) -> tuple[float, float] | None:
        """A ``(row, col)`` grid centroid as the world x, y of its cell
        centre; None when it is not a pair of numbers."""
        if not isinstance(centroid, (list, tuple)) or len(centroid) != 2:
            return None
        try:
            row = int(centroid[0])
            col = int(centroid[1])
        except (TypeError, ValueError):
            return None
        cell = self._cell_size_m
        tx, ty = self._tile_origin_xy
        return (
            float(tx) + (col + 0.5) * cell,
            float(ty) + (row + 0.5) * cell,
        )

    def _prior_surface_xy_locked(self) -> dict:
        """``{fid: (x, y)}`` for every prior row as the surface serves it,
        built once (priors do not move). Under the lock."""
        if self._prior_xy_cache is not None:
            return self._prior_xy_cache
        from maps.search.analyze_scene.geom_program import (  # noqa: PLC0415
            build_surface_from_graph,
        )

        _c, anchors = build_surface_from_graph(
            self.state.runtime_graph.nodes.items(), self._tile_origin_xy,
            float(self._cell_size_m))
        self._prior_xy_cache = {str(r["fid"]): (float(r["xy"][0]), float(r["xy"][1]))
                                for r in anchors if r.get("xy")}
        return self._prior_xy_cache

    def _world_xy_for_node_locked(self, target_fid: str) -> tuple[float, float] | None:
        """Where the fid's row says it is, the point a flight to it flies:
        a detection's ``world_xy``, else a prior's served row point, else
        the node's cell centroid (a cluster member no row serves). The
        precedence is ``geom_program.build_surface``'s. None off the graph
        or without a position. Under the lock."""
        node = self._node_for_fid_locked(target_fid)
        if node is None:
            return None
        w = node.world_xy
        if w is not None and len(w) >= 2:
            return (float(w[0]), float(w[1]))
        served = self._prior_surface_xy_locked().get(str(target_fid))
        if served is not None:
            return served
        return self._grid_centroid_to_world_xy(node.centroid)

    def node_xy(self, fid: str) -> tuple[tuple[float, float], str] | None:
        """``((x, y), "detected" | "prior")`` for a fid on the graph, the
        position as :meth:`_world_xy_for_node_locked` resolves it; None off
        the graph or without a position."""
        with self._lock:
            xy = self._world_xy_for_node_locked(fid)
            if xy is None:
                return None
            node = self._node_for_fid_locked(fid)
            return xy, ("detected" if node.provenance == "detected" else "prior")

    def _node_for_fid_locked(self, fid: str):
        """The graph node for ``fid``, or None. Under the lock."""
        return self.state.runtime_graph.nodes.get(fid)

    def _cells_xy_locked(self, target_fid: str) -> list[tuple[float, float]] | None:
        """The centres of this fid's cells in world metres, its footprint as
        the bake rasterized it; None off the graph or without cells. Under
        the lock."""
        node = self._node_for_fid_locked(target_fid)
        cells = node.cells if node is not None else None
        # cells is an (N, 2) array: its truth value raises, its length does not.
        if cells is None or len(cells) == 0:
            return None
        cell = self._cell_size_m
        tx, ty = self._tile_origin_xy
        return [(float(tx) + (int(c[1]) + 0.5) * cell, float(ty) + (int(c[0]) + 0.5) * cell) for c in cells]

    def node_cells_xy(self, target_fid: str) -> list[tuple[float, float]] | None:
        """The ground this fid stands on, as the centres of its cells in
        world metres: the area a flight to it has to hold before it has
        arrived (the flight mission's arrival rule). None off the graph or
        without cells."""
        with self._lock:
            return self._cells_xy_locked(target_fid)

    def node_coverage_stops(self, target_fid: str, spacing_m: float, *,
                            max_stops: int, minor_spacing_m: float) -> list | None:
        """Points on this fid's cells, in world metres, rounded: the stops an
        ``explore`` flies, in shape order; None off the graph or without
        cells. The stops come off the cells, never the box (a road fills
        half its box at the median) or the unclipped OSM ring.

        ``spacing_m`` is the length one stop covers and ``minor_spacing_m``
        the footprint's short side, both from the run's observation extent.
        A shape wider than ``minor_spacing_m`` on both axes tiles serpentine:
        rows along the dominant axis at that pitch, alternating direction so
        the pass stays one line. Any other shape is walked along its dominant
        axis, ends included, and one shorter than a stop is covered from its
        middle cell. Over ``max_stops`` the stops thin even-stride, ends kept.
        """
        with self._lock:
            points = self._cells_xy_locked(target_fid)
            if points is None:
                return None
            cell = self._cell_size_m
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            half = cell / 2.0
            x0, y0 = min(xs) - half, min(ys) - half
            x1, y1 = max(xs) + half, max(ys) + half
            span = max(1.0, float(spacing_m))
            if (x1 - x0) > float(minor_spacing_m) and (y1 - y0) > float(minor_spacing_m):
                pitch = max(1.0, float(minor_spacing_m))
                x_major = (x1 - x0) >= (y1 - y0)
                _maj = (lambda p: p[0]) if x_major else (lambda p: p[1])
                _mnr = (lambda p: p[1]) if x_major else (lambda p: p[0])
                lo_min = y0 if x_major else x0
                rows: dict[int, list] = {}
                for p in zip(xs, ys):
                    rows.setdefault(
                        int((_mnr(p) - lo_min) // pitch), []).append(p)
                picks = []
                for ri, ridx in enumerate(sorted(rows)):
                    row = sorted(rows[ridx], key=_maj,
                                 reverse=bool(ri % 2))
                    r_reach = abs(_maj(row[-1]) - _maj(row[0]))
                    picks.extend(_spread(row, min(int(max_stops), int(r_reach // span) + 1)))
                if len(picks) > int(max_stops):
                    picks = _spread(picks, int(max_stops))
            else:
                along = sorted(zip(xs, ys),
                               key=(lambda p: p[0])
                               if (x1 - x0) >= (y1 - y0)
                               else (lambda p: p[1]))
                reach = max(x1 - x0, y1 - y0)
                picks = _spread(along, min(int(max_stops), int(reach // span) + 1))
            on_it: list = []
            for px, py in picks:
                q = [round(px), round(py)]
                if q not in on_it:
                    on_it.append(q)
            return on_it

    def render_candidate_crops(self, requests: list[dict]) -> list[dict]:
        """The ``look`` renderer, bound to this episode's map and imagery."""
        from maps.search.render import look_render  # noqa: PLC0415

        return look_render.render_candidate_crops(
            requests,
            map_name=self._map_name,
            rgbd_dir=self._rgbd_dir,
            imagery=self.look_imagery,
        )

    @property
    def state(self) -> _State:
        if self._state is None:
            raise RuntimeError("SearchLoop.start must be called before update.")
        return self._state

    def boost_queue(self, concepts, where) -> dict:
        """The boost channel the mission's ``look_for(..., at=)`` calls."""
        with self._lock:
            return self.state.perception_director.boost_queue(concepts, where)

    def set_retro_perceiver(self, fn) -> None:
        """Install the retrospective executor: re-render a flown stop's
        recorded pose, re-segment it with the boost's classes, fold it, and
        return the delta."""
        with self._lock:
            self.state.perception_director.retro_perceiver = fn

    def start_flight(self) -> None:
        """Put the queued boosts' classes on the coming flight's segmenter."""
        with self._lock:
            self.state.perception_director.start_flight()

    def record_flown_stop(self, flown: dict) -> None:
        """Log a perceived flight's stop for the boost to address, and
        re-push the watchlist."""
        with self._lock:
            self.state.perception_director.record_flown(flown)

    def declare_mission_concepts(self, classes: list[str]) -> str | None:
        """Set the segmenter's classes to the declared list of classes and
        diff the director's watchlist to it. Returns the error the mission
        serves when the setter failed, else None."""
        # The classes SAM segments, canonical, first seen first; a placeholder
        # class is no class.
        union = list(dict.fromkeys(
            c for c in (canon_concept(x) for x in classes)
            if c and c not in _NULL_CONCEPTS))
        setter_err: str | None = None
        with self._lock:
            if self._sam_setter is not None:
                try:
                    self._sam_setter(union)
                except Exception as e:  # noqa: BLE001
                    # never a clean receipt over a segmenter that kept its
                    # old classes
                    log.warning("sam concept setter failed: %s", e)
                    setter_err = str(e)
            d = self.state.perception_director
            adds = [c for c in union if c not in d.watchlist]
            rems = [c for c in d.watchlist if c not in union]
            if adds or rems:
                d.watchlist_apply(adds, rems)
        if setter_err:
            return (f"the segmenter did not take this declaration "
                    f"({setter_err}); say it again")
        return None

    def update_perception(self, detections: list, projected_masks: dict[int, object]) -> list[dict]:
        """Fold one flight's detections into the runtime graph: tracked ones
        with a projected footprint through the tracked upsert, the rest
        centroid-only. Returns the per-step detection records the recorder
        writes."""
        from maps.sws.graph.runtime import (  # noqa: PLC0415
            upsert_detected_centroid_only,
            upsert_tracked_detected,
        )

        with self._lock:
            s = self.state
            bounds_min, cell = self._tile_origin_xy, self._cell_size_m
            # The world tracker stamps each detection's stable cross-frame
            # ``track_id`` ('wt_<n>'), the upsert's sole re-id key.
            self._world_tracker.update(detections)
            upsert_ids: dict[int, str] = {}
            for i, tp in projected_masks.items():
                det = detections[i]
                if not (det.metadata or {}).get("track_id"):
                    continue
                upsert_ids[i] = upsert_tracked_detected(
                    s.runtime_graph,
                    track_id=self._detection_id(det),
                    base_node=_detection_to_mapobject(det, bounds_min, cell),
                    # Sparse occupied cells, never a dense tile raster.
                    rows=tp.rows,
                    cols=tp.cols,
                )

            step_detections: list[dict] = []
            for i, det in enumerate(detections):
                if i not in upsert_ids:
                    upsert_detected_centroid_only(
                        s.runtime_graph, _detection_to_mapobject(det, bounds_min, cell))
                det_record = {
                    "world_xy": list(det.world_xy),
                    "label": det.label,
                    "confidence": float(det.confidence),
                    "source_tier": det.source_tier,
                    "attributes": dict(det.attributes or {}),
                    # tile_cells feeds the upsert above; str() of an array truncates it.
                    "metadata": {k: _json_safe(v) for k, v in (det.metadata or {}).items()
                                 if k != "tile_cells"},
                }
                if det.mask is not None:
                    mask_record = encode_mask_record(det.mask.crop, bbox_rcrc=det.mask.bbox)
                    if mask_record is not None:
                        det_record.update(mask_record)
                step_detections.append(det_record)

            # THIS flight's tracked detections, by the fids the analyze
            # surface serves, for the flight report; replaced whole every
            # flight. xy is the detection's own, durable if a later dedup
            # re-homes the fid.
            seen: dict[str, dict] = {}
            for i, fid in sorted(upsert_ids.items()):
                det = detections[i]
                seen.setdefault(fid, {
                    "fid": fid,
                    "class": str(det.label or ""),
                    "xy": [round(float(det.world_xy[0]), 1), round(float(det.world_xy[1]), 1)],
                    # The detector's own score: the report heads each class
                    # with its best.
                    "conf": round(float(det.confidence or 0.0), 3),
                })
            s.last_flight_detections = list(seen.values())
            return step_detections

    def last_flight_detections(self) -> list[dict]:
        """Copies of the rows the last perception update folded, the flight
        report's source."""
        with self._lock:
            return [dict(d) for d in self.state.last_flight_detections]

    def upsert_prior_percept(self, fid: str, attrs: dict, *, px: int = 0) -> bool:
        """Merge attributes perceived inside a prior's projected footprint
        (e.g. ``{"color": "gray", "color_rgb": "128,130,131"}``) onto its
        node. Never raises: the per-frame enrichment calls it inline."""
        try:
            from maps.sws.graph.runtime import (
                upsert_prior_percept as _upsert_prior_percept,
            )

            with self._lock:
                return _upsert_prior_percept(
                    self.state.runtime_graph, str(fid), dict(attrs or {}), px=px
                )
        except Exception:  # noqa: BLE001
            log.debug("upsert_prior_percept failed for %s", fid, exc_info=True)
            return False

    def _tool_ctx_for_examiner(self):
        """The :class:`ToolContext` over the live graph and the tile frame."""
        from maps.search.analyze_scene.tools import ToolContext

        return ToolContext(
            grid_shape=(int(self._grid.height), int(self._grid.width)),
            graph=self.state.runtime_graph,
            tile_origin_xy=self._tile_origin_xy,
            cell_size_m=self._cell_size_m,
        )

    @staticmethod
    def _detection_id(det) -> str:
        """The world tracker's ``wt_<n>`` id, else ``label@x,y``, which
        collides only on co-location. The segmenter's frame-local
        ``sam3_1_img_<n>`` id is never the key: it would match the next
        stop's unrelated detection n."""
        track = (det.metadata or {}).get("track_id")
        if isinstance(track, str) and track.startswith("wt_"):
            return track
        return f"{det.label or '?'}@{float(det.world_xy[0]):.0f},{float(det.world_xy[1]):.0f}"


def _spread(seq: list, k: int) -> list:
    """``k`` (clamped to 1 .. ``len(seq)``) items of ``seq`` at an even stride,
    both ends kept; one item is the middle one, since an end would see half
    the shape."""
    k = max(1, min(k, len(seq)))
    if k == 1:
        return [seq[(len(seq) - 1) // 2]]
    stride = (len(seq) - 1) / (k - 1)
    return [seq[int(round(i * stride))] for i in range(k)]


def _detection_to_mapobject(det, bounds_min, cell_size_m):
    """One detection as a graph node without cells (the tracked upsert stamps
    them): its centroid (the mean of its stamped tile cells, else the cell
    under its world point), contour, attributes and world point."""
    meta = det.metadata or {}
    raw_cells = meta.get("tile_cells")
    cells: np.ndarray | None = None
    if raw_cells is not None:
        try:
            cells_arr = np.asarray(raw_cells, dtype=np.int32).reshape(-1, 2)
            if cells_arr.size > 0:
                cells = cells_arr
        except (TypeError, ValueError):
            cells = None

    # The footprint polygon in world metres the ortho projection stamped;
    # the upsert re-stamps it on repeat sightings.
    raw_contour = meta.get("world_contour")
    world_contour: list[list[float]] | None = None
    if raw_contour is not None:
        try:
            wc = [[float(p[0]), float(p[1])] for p in raw_contour]
            if len(wc) >= 3:
                world_contour = wc
        except (TypeError, ValueError, IndexError):
            world_contour = None

    col = math.floor((det.world_xy[0] - bounds_min[0]) / cell_size_m)
    row = math.floor((det.world_xy[1] - bounds_min[1]) / cell_size_m)
    if cells is not None:
        centroid = (
            int(round(float(cells[:, 0].mean()))),
            int(round(float(cells[:, 1].mean()))),
        )
    else:
        centroid = (row, col)
    semantic_attributes: dict[str, dict[str, int]] = {
        key: {value: 1} for key, value in (det.attributes or {}).items()
    }
    # ``color_px`` (the mask's in-view pixel count) becomes best_view_px, not
    # an attribute: the upsert replaces colour only from a larger view.
    best_view_px = 0
    _bv_bucket = semantic_attributes.pop("color_px", None)
    if _bv_bucket:
        try:
            best_view_px = int(next(iter(_bv_bucket)))
        except (TypeError, ValueError):
            best_view_px = 0
    label = str(det.label or "").strip()
    if label:
        label_bucket = semantic_attributes.setdefault("sam_label", {})
        label_bucket[label] = label_bucket.get(label, 0) + 1
    return MapObject(
        class_name=label or "object",
        name=det.label,
        tags={},
        centroid=centroid,
        provenance="detected",
        best_view_px=best_view_px,
        # the tracked upsert stamps the node's own cells
        semantic_attributes=semantic_attributes,
        cells=None,
        world_contour=world_contour,
        # the sub-cell position build_surface prefers to the cell centroid
        world_xy=(float(det.world_xy[0]), float(det.world_xy[1])),
    )


def _json_safe(v):
    if isinstance(v, (str, int, float, bool, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)
