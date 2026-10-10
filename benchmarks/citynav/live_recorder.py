"""Per-step writer for live / replay dashboarding.

One recorder per episode directory. A step is one line, a detection is one
line, and the engine's own sidecars live under ``flight/``:

    {episode_dir}/
        steps.jsonl              # one line per step, the payload below
        detections.jsonl         # one line per detection, {**detection, "step": int}
        flight/
            heartbeat.json       # {ts, episode_id, status, step} - fresh each step
            .advance_state.json  # {episode_id, split} - the active episode
            manifest.json        # split, episodes[], current_episode, status
            step_{n:04d}_graph.json   # the graph-delta sidecar, when tracked
            step_timings.jsonl   # {step, ts, timings_s}

Step row schema (JSON, no numpy types):

    {
      "step": int,
      "episode_id": str,
      "pose_xy": [x, y],
      "altitude_m": float,
      "heading_deg": float,
      "frame_px": int | null,   # square edge of the frame perception read;
                                # the pixel space every mask bbox lives in
      "visited_xy": [[x, y], ...],
      "fov_corners_xy": [[x, y], [x, y], [x, y], [x, y]] | null,
      "compile": {
        "wall_s": float,
        "parsed_instruction": str,
        "alive_rids": [int, ...],
        "broken_rids": [int, ...],
      } | null,
      "graph_sidecar": str | null,  # "flight/step_{n:04d}_graph.json",
                                    # relative to the episode dir
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
No pixels are written per step —
the block raster is a bake asset the viewer holds locally. Writing pixels
instead cost 3.8 MB a step, on disk and on the sync wire.

The JSONL files are append-and-flush, so a reader sees at most one torn final
line; the JSON files are tempfile-then-rename so a polling reader never sees
a half-written one.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, default=_json_fallback)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


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
    split: str

    def __post_init__(self) -> None:
        self.episode_dir = Path(self.episode_dir)
        self.flight = self.episode_dir / "flight"
        self.flight.mkdir(parents=True, exist_ok=True)
        self._current_step: int = 0
        self._status: str = "starting"
        # Run-meta surfaced on every manifest write so the SPA can switch
        # layout based on the run's executor mode at any point in its
        # lifecycle. Default is None; the runner calls set_run_meta() at
        # startup with the resolved cfg.
        self._compliant_parallel: bool | None = None
        self._planner: str | None = None
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._write_manifest()
        _atomic_write_json(
            self.flight / ".advance_state.json",
            {"episode_id": self.episode_id, "split": self.split},
        )
        # Both files exist from the start: a mission that flies nothing has
        # zero steps, which an empty file says and an absent one does not.
        for name in (STEPS_JSONL, DETECTIONS_JSONL):
            (self.episode_dir / name).touch()

    def set_run_meta(
        self,
        *,
        compliant_parallel: bool,
        planner: str,
    ) -> None:
        """Record the run's executor mode so the SPA can pick its layout.

        The SPA's SensorView reads ``compliant_parallel`` off the
        manifest to switch between the photo-gallery layout (parallel
        sweep: every tour stop as a row, 3 columns of RGB / depth / SAM
        imagery) and the live video-timeline layout (sequential
        discrete flight).
        """
        self._compliant_parallel = bool(compliant_parallel)
        self._planner = str(planner)
        self._write_manifest()
        self._write_heartbeat()
        # Background ticker keeps the heartbeat fresh during long
        # compile turns so the dashboard stays in "flight" phase.
        if self._heartbeat_thread is None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, name="LiveRecorderHeartbeat", daemon=True
            )
            self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(5.0):
            try:
                self._write_heartbeat()
            except Exception:
                # File-system hiccup — try again next tick.
                pass

    def stop(self) -> None:
        """Stop the background heartbeat ticker."""
        self._stop_event.set()

    def _write_heartbeat(self) -> None:
        payload = {
            "ts": time.time(),
            "episode_id": self.episode_id,
            "status": self._status,
            "step": self._current_step,
        }
        _atomic_write_json(self.flight / "heartbeat.json", payload)

    # --- Manifest ---

    def set_status(self, status: str) -> None:
        self._status = status
        self._write_manifest()
        self._write_heartbeat()

    def _write_manifest(self) -> None:
        payload = {
            "split": self.split,
            "status": self._status,
            "episodes": [self.episode_id],
            "current_episode": self.episode_id,
            "current_step": self._current_step,
            "compliant_parallel": self._compliant_parallel,
            "planner": self._planner,
        }
        _atomic_write_json(self.flight / "manifest.json", payload)

    # --- Per-step rows ---

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
        compile_info: dict[str, Any] | None,
        wall_t_monotonic: float,
        frame_px: int | None = None,
        sensor: str | None = None,
        obs_class_histogram: list[dict[str, Any]] | None = None,
        detected_node_class_histogram: list[dict[str, Any]] | None = None,
        planned_route: list[tuple[float, float]] | None = None,
        replan_triggered: bool = False,
        graph_payload: dict[str, Any] | None = None,
        sample_rate_hz: float | None = None,
        discrete_action: str | None = None,
        discrete_steps_emitted: int = 0,
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
            "compile": compile_info,
            "wall_t_monotonic": float(wall_t_monotonic),
            "obs_class_histogram": obs_class_histogram or [],
            # Per-class counts of detected runtime-graph nodes. Verifies the
            # two-layer perception contract (``car`` count layer + ``gray
            # car`` color overlay appear as separate classes).
            "detected_node_class_histogram": detected_node_class_histogram or [],
            "planned_route": (
                [list(p) for p in planned_route] if planned_route else []
            ),
            "replan_triggered": bool(replan_triggered),
            # The graph-delta sidecar, relative to the episode dir, so a
            # reader fetches it without scanning. ``None`` when no graph
            # tracking is active.
            "graph_sidecar": (
                f"flight/step_{step:04d}_graph.json"
                if graph_payload is not None
                else None
            ),
            # Flight-mode frame cadence. When set, downstream video
            # renderers can default FPS to this value so the playback
            # matches the simulated camera rate.
            "sample_rate_hz": (
                float(sample_rate_hz) if sample_rate_hz is not None else None
            ),
            # CityNav six-action contract instrumentation. ``discrete_action``
            # is one of {move_forward, turn_left, turn_right, ascend,
            # descend, stop} when the discrete flight sim emitted an
            # action this step; ``None`` in continuous-sampling mode.
            # ``discrete_steps_emitted`` is the cumulative count and tracks
            # the CityNav step budget directly.
            "discrete_action": discrete_action,
            "discrete_steps_emitted": int(discrete_steps_emitted),
        }
        _append_jsonl(self.episode_dir / STEPS_JSONL, payload)
        for det in detections:
            _append_jsonl(self.episode_dir / DETECTIONS_JSONL, {**det, "step": int(step)})
        if graph_payload is not None:
            _atomic_write_json(self.flight / f"step_{step:04d}_graph.json", graph_payload)
        self._current_step = step
        self._write_manifest()
        self._write_heartbeat()

    def append_step_timing(
        self,
        *,
        step: int,
        timings_s: dict[str, float],
    ) -> None:
        """Append one row to ``flight/step_timings.jsonl``.

        Each row is ``{"step": int, "ts": float, "timings_s": {...}}`` so
        the profile rollup can attribute wall-clock seconds to perception
        / planner / writer / compile-drain phases without having to
        regex-parse run.log.
        """
        _append_jsonl(
            self.flight / "step_timings.jsonl",
            {
                "step": int(step),
                "ts": time.time(),
                "timings_s": {k: float(v) for k, v in timings_s.items()},
            },
        )
