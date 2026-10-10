"""Episode-level harness state for the stepwise CityNav runner.

Pose, visit history, episode bookkeeping, and a reference to the prior
graph live here. Belief, the runtime graph, scout / lock state, paint
layers, parse-compile artifacts, and the rest of the autonomy state
machine live on the ``SearchLoop`` from ``maps.search``; the runner
mirrors a frozen ``SearchSnapshot`` onto ``belief_snapshot`` each step
so the recorder and the post-episode scoring can read it through one
indirection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from maps.search import SearchSnapshot
from maps.perception.geometry.sensor_spec import SensorSpec
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
    # Camera spec resolved from cfg.sensor at episode start. The ortho crop
    # is orthographic and carries no sensor, so this is the planner-facing
    # geometry (intrinsics, ranges, and the description the reasoner reads),
    # not something the observation is projected through.
    sensor_spec: SensorSpec | None = None
    # Mirror of ``SearchLoop.snapshot()`` written each step. Recorder and
    # scoring read autonomy state through this field.
    belief_snapshot: SearchSnapshot | None = None
    def record_visit(self) -> None:
        self.visited_xy.append(self.pose_xy)
