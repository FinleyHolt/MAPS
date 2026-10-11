"""Per-step writer: where the drone was each step, and what it detected.

One recorder per episode directory writes two append-and-flush JSONL files
(a reader sees at most one torn final line), both present from the start:

- ``steps.jsonl``, one line per step: ``step``, ``episode_id``, ``pose_xy``,
  ``altitude_m``, ``heading_deg``, ``frame_px``, ``sensor``, ``visited_xy``,
  ``fov_corners_xy``, ``wall_t_monotonic``. Pose, altitude, heading,
  ``frame_px`` and ``sensor`` are a complete render spec over the bake's
  ortho, so no pixels are written.
- ``detections.jsonl``, one line per detection the step's perception fold
  produced (``maps.search.search_loop``'s record, masks packed) plus ``step``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_fallback(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _append_jsonl(path: Path, record: Any) -> None:
    line = json.dumps(record, default=_json_fallback) + "\n"
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
    except OSError:
        pass


STEPS_JSONL = "steps.jsonl"
DETECTIONS_JSONL = "detections.jsonl"


@dataclass
class LiveRecorder:
    episode_dir: Path
    episode_id: str

    def __post_init__(self) -> None:
        self.episode_dir = Path(self.episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # A mission that flies nothing has zero steps: an empty file says so.
        for name in (STEPS_JSONL, DETECTIONS_JSONL):
            (self.episode_dir / name).touch()

    def write_step(
        self,
        *,
        step: int,
        pose_xy: tuple[float, float],
        altitude_m: float,
        heading_deg: float,
        visited_xy: list[tuple[float, float]],
        fov_corners_xy: list[tuple[float, float]] | None,
        detections: list[dict[str, Any]],
        wall_t_monotonic: float,
        frame_px: int | None = None,
        sensor: str | None = None,
    ) -> None:
        payload = {
            "step": step,
            "episode_id": self.episode_id,
            "pose_xy": list(pose_xy),
            "altitude_m": float(altitude_m),
            "heading_deg": float(heading_deg),
            # The frame's square edge, the pixel space of every ``pixel_bbox_rcrc``;
            # with a ``sensor`` (registry name) fixing width x height, its height.
            "frame_px": int(frame_px) if frame_px is not None else None,
            "sensor": sensor,
            "visited_xy": [list(p) for p in visited_xy],
            "fov_corners_xy": (
                [[float(c[0]), float(c[1])] for c in fov_corners_xy]
                if fov_corners_xy
                else None
            ),
            "wall_t_monotonic": float(wall_t_monotonic),
        }
        _append_jsonl(self.episode_dir / STEPS_JSONL, payload)
        for det in detections:
            _append_jsonl(self.episode_dir / DETECTIONS_JSONL, {**det, "step": int(step)})
