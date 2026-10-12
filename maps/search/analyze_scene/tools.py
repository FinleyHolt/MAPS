"""The tool context: the world the reasoner's analyze programs query."""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.sws.graph.model import ObjectGraph


@dataclass
class ToolContext:
    """The shared state every tool queries: the one world-state graph (prebake
    seed plus every in-flight detection; nodes carry ``provenance``
    "prebaked" / "detected") and the tile frame it sits in."""

    grid_shape: tuple[int, int]  # (H, W)
    graph: ObjectGraph = field(repr=False)
    pose_xy: tuple[float, float] | None = None  # the drone's world x, y; None when unknown
    tile_origin_xy: tuple[float, float] = (0.0, 0.0)
    cell_size_m: float = 0.0  # required; caller passes the tile's true value
