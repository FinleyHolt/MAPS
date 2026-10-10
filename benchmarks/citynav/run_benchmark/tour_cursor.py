"""``flight/tour_cursor.json``: the flown-stop record."""

from __future__ import annotations

from maps.search.parse_compile.tour_state import FeatureTour
from pathlib import Path
import json
import os
import logging

log = logging.getLogger("run_benchmark")


def _write_tour_cursor(tour: FeatureTour) -> None:
    """Persist the flown stops to ``<workdir>/flight/tour_cursor.json``
    (atomic via temp + replace): each stop's id and xy, in flight order, the
    record the flight readers (``benchmarks.flight.measures`` and
    ``replay``) take a run's flights from. No-op when
    ``SEARCH_COMPILE_WORKDIR`` is not set (e.g. unit-test invocations).
    """
    workdir_env = os.environ.get("SEARCH_COMPILE_WORKDIR")
    if not workdir_env:
        return
    try:
        flight_dir = Path(workdir_env) / "flight"
        flight_dir.mkdir(parents=True, exist_ok=True)
        path = flight_dir / "tour_cursor.json"
        payload = {
            "stops_visited": [
                {"stop_id": vp.fid, "xy": list(vp.xy)} for vp in tour.viewpoints
            ],
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)
    except Exception:
        log.debug("tour_cursor write failed; non-fatal", exc_info=True)
