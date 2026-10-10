"""Per-step writer: where the drone was each step, and what it detected.

One recorder per episode directory:

    {episode_dir}/
        steps.jsonl              # one line per step, the payload below
        detections.jsonl         # one line per detection, {**detection, "step": int}

Step row schema (JSON, no numpy types):

    {
      "step": int,
      "episode_id": str,
      "pose_xy": [x, y],
      "altitude_m": float,
      "heading_deg": float,
      "frame_px": int | null,   # square edge of the frame perception read;
                                # the pixel space every mask bbox lives in
      "sensor": str | null,
      "visited_xy": [[x, y], ...],
      "fov_corners_xy": [[x, y], [x, y], [x, y], [x, y]] | null,
      "wall_t_monotonic": float,
    }

Detection row schema, one per detection the step's perception fold produced
(the record ``maps.search.search_loop`` builds, plus ``step``):

    {
      "step": int,
      "world_xy": [x, y], "label": str, "confidence": float,
      "class_hint": str | null, "source_tier": str | null,
      "description": str | null, "attributes": {...}, "metadata": {...},
      "pixel_bbox_rcrc": [r0, c0, r1, c1], "mask_shape": [h, w],
      "mask_zpacked_b64": str,
    }

A step records where the drone was, not what it saw. ``pose_xy``,
``altitude_m``, ``heading_deg`` and ``frame_px`` are a complete render spec
over the one hole-free ortho, and the detection masks travel packed in the
detection rows, so a reader rebuilds rgb and the SAM overlay from the bake.
No pixels are written per step.

The JSONL files are append-and-flush, so a reader sees at most one torn final
line.
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
        # Both files exist from the start: a mission that flies nothing has
        # zero steps, which an empty file says and an absent one does not.
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
            # Square edge of the frame perception read. With pose, altitude and
            # heading this is the whole render spec, and it is the pixel space
            # every detection's ``pixel_bbox_rcrc`` is measured in — the audit
            # cannot place a mask without it. On a sensor-footprint run,
            # ``sensor`` (a registry name) joins the spec and fixes the frame's
            # true width × height; frame_px then records the frame HEIGHT.
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
