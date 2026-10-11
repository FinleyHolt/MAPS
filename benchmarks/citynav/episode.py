"""Episode-level harness state for the stepwise CityNav runner: pose,
visit history, the prior graph and the bound commit. The runtime graph
lives on the ``SearchLoop`` from ``maps.search``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.sws.graph.model import ObjectGraph

__all__ = [
    "EpisodeState",
]


@dataclass
class EpisodeState:
    pose_xy: tuple[float, float]
    altitude_m: float
    graph: ObjectGraph  # frozen prior graph (prebaked OSM/CityRefer)
    visited_xy: list[tuple[float, float]] = field(default_factory=list)
    step_idx: int = 0
    heading_deg: float = 0.0  # 0 = facing north (+y); derived from motion vector.
    # The last flight's resolved point, bound when the mission ends; None
    # when nothing flew. The trajectory replay flies to it.
    confirmed_target_xy: tuple[float, float] | None = None

    def record_visit(self) -> None:
        self.visited_xy.append(self.pose_xy)
