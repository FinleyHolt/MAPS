"""The query context: the world the model's programs query."""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.world_state.graph import ObjectGraph


@dataclass
class QueryContext:
    """The shared state every program queries: the one world-state graph (prebake
    seed plus every in-flight detection; nodes carry ``provenance``
    "prebaked" / "detected") and the tile frame it sits in."""

    grid_shape: tuple[int, int]  # (H, W)
    graph: ObjectGraph = field(repr=False)
    tile_origin_xy: tuple[float, float]
    cell_size_m: float
    pose_xy: tuple[float, float] | None = None  # the drone's world x, y; None when unknown
