"""The run's ledger and the attempt's row.

    <run>/episodes.jsonl                    one line per ATTEMPT, appended by the
                                            run-root OWNER (the fleet driver, the
                                            batch loop) when the attempt lands
    <run>/[w<id>/]<episode>/episode.json    the runner's row for that attempt,
                                            overwritten atomically as it moves

``episode.json`` holds what the runner knows: ``status`` is ``flying`` from
the moment the directory opens (the pre-flight floor carries ``provisional:
true``) until the attempt lands as ``ok`` or ``infra``. A ledger row is that
row plus what only the owner knows -- ``workdir`` relative to the ledger's
directory, ``worker_id``, ``oom_retries`` -- and the owner overrides
``status`` on its own kill paths (``timed_out``, ``infra``). A driver-killed
attempt leaves its ``episode.json`` at ``flying``; the ledger row is the
verdict.
"""

from __future__ import annotations

import fcntl
import json
import math
import threading
from pathlib import Path
from typing import Any

EPISODES_JSONL = "episodes.jsonl"
EPISODE_JSON = "episode.json"

STATUS_FLYING = "flying"
STATUS_OK = "ok"
STATUS_TIMED_OUT = "timed_out"
STATUS_INFRA = "infra"

# Threads within one owner serialize here; owners in other processes
# (lanes sharing a root) serialize on the flock.
_append_lock = threading.Lock()


def strict(value: Any) -> Any:
    """``value`` with every non-finite float replaced by None.

    The rows are a dataset other people read with whatever parser they
    have, and ``Infinity``/``NaN`` are Python's JSON dialect, not JSON: a
    sensor's ``max_range_m`` of ``inf`` broke ``jq`` on the first row it
    met. Absent is what an unbounded range means to a reader anyway.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict(v) for v in value]
    return value


def append_row(root: Path, row: dict) -> None:
    line = json.dumps(strict(row), allow_nan=False)
    with _append_lock, (Path(root) / EPISODES_JSONL).open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def upsert_row(root: Path, row: dict) -> None:
    """Land ``row`` as the ledger's ONE row for its ``run_no``, replacing an
    earlier row for the same run.

    A field mission's row lands the moment the model reports rather than
    only when its seat is cleared, so a station restarted over a parked run
    does not lose it. That run flies again on the operator's next tasking
    and lands again, and every reader wants ONE row per run: the station
    numbers the next run off the ledger's LENGTH, and the analysis readers
    count an episode once.

    ``run_no`` is the run's number in THIS ledger, which is what tells two
    runs apart inside one: ``run_id`` is the root's own name, shared by
    every run that landed in it. The file is rewritten under the lock the
    append takes -- a sortie's ledger is one row per run, and its readers
    load it whole anyway. A row naming no run is appended, having nothing to
    be the same run as.
    """
    run_no = row.get("run_no")
    if run_no is None:
        append_row(root, row)
        return
    line = json.dumps(strict(row), allow_nan=False)
    path = Path(root) / EPISODES_JSONL
    with _append_lock, path.open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            kept: list[str] = []
            replaced = False
            for text in f:
                text = text.strip()
                if not text:
                    continue
                try:
                    same = json.loads(text).get("run_no") == run_no
                except ValueError:
                    same = False  # an unreadable row is kept, never dropped
                kept.append(line if same else text)
                replaced = replaced or same
            if not replaced:
                kept.append(line)
            f.seek(0)
            f.truncate()
            f.write("\n".join(kept) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_episode_json(episode_dir: Path, row: dict) -> None:
    # Imported here: live_recorder pulls numpy, and the fleet driver and the
    # launcher import this module before any episode needs it.
    from benchmarks.citynav.live_recorder import _atomic_write_json

    _atomic_write_json(Path(episode_dir) / EPISODE_JSON, strict(row))


def read_episode_json(episode_dir: Path) -> dict | None:
    try:
        return json.loads((Path(episode_dir) / EPISODE_JSON).read_text())
    except (OSError, ValueError):
        return None
