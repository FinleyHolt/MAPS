"""``flight/tour_cursor.json``: the executor's flown-stop sidecar the SPA polls."""

from __future__ import annotations

from maps.search.parse_compile.tour_state import FeatureTour
import json
import os
import logging

log = logging.getLogger("run_benchmark")


def _write_tour_cursor(
    tour: FeatureTour,
    step_idx: int,
    pose_xy: tuple[float, float] | None = None,
    heading_deg: float | None = None,
) -> None:
    """Persist the executor's tour cursor for the dashboard.

    Writes ``<workdir>/flight/tour_cursor.json`` (atomic via temp +
    replace). The SPA polls this file to render the executor's position
    on the tour polyline; ``pose_xy`` + ``heading_deg`` carry the live
    drone pose so the dashboard can draw a drone marker. No-op when
    ``SEARCH_COMPILE_WORKDIR`` is not set (e.g. unit-test invocations).
    """
    workdir_env = os.environ.get("SEARCH_COMPILE_WORKDIR")
    if not workdir_env:
        return
    try:
        from pathlib import Path as _Path

        flight_dir = _Path(workdir_env) / "flight"
        flight_dir.mkdir(parents=True, exist_ok=True)
        path = flight_dir / "tour_cursor.json"
        payload: dict = {
            "cursor": int(tour.cursor),
            "step_idx": int(step_idx),
            "stops_total": len(tour.viewpoints),
            "stops_visited": [
                {"stop_id": vp.fid, "xy": list(vp.xy), "prob": float(vp.prob)}
                for vp in tour.viewpoints[: tour.cursor]
            ],
            "stops_remaining": [
                {"stop_id": vp.fid, "xy": list(vp.xy), "prob": float(vp.prob)}
                for vp in tour.viewpoints[tour.cursor :]
            ],
        }
        if pose_xy is not None:
            drone_pose = {
                "x": float(pose_xy[0]),
                "y": float(pose_xy[1]),
                "heading_deg": float(heading_deg) if heading_deg is not None else 0.0,
            }
            payload["drone_pose"] = drone_pose
            payload["robot_pose"] = drone_pose
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except Exception:
        log.debug("tour_cursor write failed; non-fatal", exc_info=True)
