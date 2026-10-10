"""The run's ledger and the attempt's row.

    <run>/episodes.jsonl                    one line per ATTEMPT, appended by the
                                            run-root OWNER (the fleet driver, the
                                            batch loop) when the attempt lands
    <run>/[w<id>/]<episode>/episode.json    the runner's row for that attempt,
                                            overwritten atomically as it moves

``episode.json`` holds what the runner knows once the attempt lands: ``ok``
or ``infra``. A ledger row is that row plus what only the owner knows --
``workdir`` relative to the ledger's directory, ``worker_id`` -- and the owner overrides ``status`` on its own kill paths
(``timed_out``, ``infra``). A driver-killed attempt leaves no
``episode.json``; the ledger row is the verdict.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import threading
from pathlib import Path
from typing import Any

EPISODES_JSONL = "episodes.jsonl"
EPISODE_JSON = "episode.json"

STATUS_OK = "ok"
STATUS_TIMED_OUT = "timed_out"
STATUS_INFRA = "infra"

# Threads within one owner serialize here; the flock serializes against a
# reader or a station in another process.
_append_lock = threading.Lock()

# The run root's lock file, held exclusively by the run that owns the root.
ROOT_LOCK = ".bench.lock"


class RootHeld(RuntimeError):
    """Another run owns this root, or its filesystem cannot lock."""


def hold_root(root: Path):
    """Take ``root``'s exclusive lock for this run's lifetime and return the
    open file holding it (the flock lifts when it closes or the process
    ends, however it ended).

    Two runs into one root fly the same episodes twice, and the pick lets a
    later ``ok`` supersede an earlier miss. Raises ``RootHeld`` naming the
    holder, or naming the filesystem when it cannot flock.
    """
    lock = (Path(root) / ROOT_LOCK).open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        holder = lock.read().strip() or "unknown"
        lock.close()
        raise RootHeld(
            f"another run holds {root} ({holder}); two runs into one root fly "
            "the same episodes twice. Wait for it to end, or give this one its "
            "own root."
        ) from None
    except OSError as exc:
        lock.close()
        raise RootHeld(
            f"{root} cannot be locked ({exc}); a run root needs flock: local "
            "disk, or a network filesystem mounted with locking."
        ) from None
    lock.seek(0)
    lock.truncate()
    lock.write(f"pid {os.getpid()} on {os.uname().nodename}\n")
    lock.flush()
    return lock


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
    """Append one attempt to the run's ledger, under an exclusive flock.

    A writer killed mid-append leaves a cut last line with no newline; the
    next row would glue onto it and be lost with it. So a ledger that does
    not end in a newline gets one first: the cut line stays undecodable and
    this row reads whole.
    """
    line = json.dumps(strict(row), allow_nan=False)
    with _append_lock, (Path(root) / EPISODES_JSONL).open("ab+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            size = os.fstat(f.fileno()).st_size
            if size and os.pread(f.fileno(), 1, size - 1) != b"\n":
                f.write(b"\n")
            f.write((line + "\n").encode("utf-8"))
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
