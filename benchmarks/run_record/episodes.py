"""The run's ledger and the attempt's row.

``<run>/episodes.jsonl`` holds one line per attempt, appended by the run root's
owner when the attempt lands; ``<run>/[w<id>/]<episode>/episode.json`` is the
runner's row for it (``ok`` or ``infra``). The ledger row adds what only the
owner knows (``workdir``, ``worker_id``) and the owner's own verdicts
(``timed_out``, ``infra``); a driver-killed attempt has only the ledger row.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

EPISODES_JSONL = "episodes.jsonl"
EPISODE_JSON = "episode.json"

STATUS_OK = "ok"
STATUS_TIMED_OUT = "timed_out"
STATUS_INFRA = "infra"

#: The run root's manifest; an attempt's log in its workdir; the flown tour's
#: cursor in its workdir.
MANIFEST_NAME = "run_manifest.json"
RUN_LOG = "run.log"
TOUR_CURSOR = "flight/tour_cursor.json"

#: What the driver hands each attempt's environment: its own pid (the runner
#: exits when it is gone) and the attempt's wall-clock deadline, an epoch (the
#: mission's backstop).
PARENT_PID_ENV = "CITYNAV_BENCH_PARENT_PID"
DEADLINE_ENV = "CITYNAV_EPISODE_DEADLINE_EPOCH"

# Threads serialize here; the flock serializes against other processes.
_append_lock = threading.Lock()

# The run root's lock file, held exclusively by the run that owns the root.
ROOT_LOCK = ".bench.lock"


class RootHeld(RuntimeError):
    """Another run owns this root, or its filesystem cannot lock."""


def hold_root(root: Path):
    """Take ``root``'s exclusive lock and return the open file holding it
    (the flock lifts when it closes or the process ends). Raises ``RootHeld``
    naming the holder, or the filesystem when it cannot flock.
    """
    import fcntl  # POSIX only, and this module is imported on Windows

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
    """``value`` with every non-finite float replaced by None, so the rows
    are strict JSON (``Infinity``/``NaN`` are Python's dialect)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: strict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict(v) for v in value]
    return value


@contextlib.contextmanager
def _locked_ledger(path: Path, mode: str):
    """``path`` open in ``mode`` under an exclusive flock, released on exit.
    :func:`upsert_row` replaces the file, so a lock that lands on an inode
    another writer has since replaced is dropped and taken on the current one."""
    import fcntl  # POSIX only, and this module is imported on Windows

    while True:
        f = path.open(mode)
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            current = os.stat(path).st_ino == os.fstat(f.fileno()).st_ino
        except FileNotFoundError:
            current = False
        if current:
            break
        f.close()
    try:
        yield f
    finally:
        f.close()


def append_row(root: Path, row: dict) -> None:
    """Append one attempt to the run's ledger, under an exclusive flock. A
    ledger cut mid-line by a killed writer gets a newline first, so this row
    reads whole."""
    line = json.dumps(strict(row), allow_nan=False)
    with _append_lock, _locked_ledger(Path(root) / EPISODES_JSONL, "ab+") as f:
        size = os.fstat(f.fileno()).st_size
        if size and os.pread(f.fileno(), 1, size - 1) != b"\n":
            f.write(b"\n")
        f.write((line + "\n").encode("utf-8"))
        f.flush()


def upsert_row(root: Path, row: dict) -> None:
    """Land ``row`` as the ledger's one row for its ``run_no`` (the run's
    number in this ledger), replacing an earlier row for the same run.

    A field mission's row lands when the model reports and again when the
    run flies on, and the readers count a run once. The new ledger is written
    beside the old and renamed over it under the append's lock, so a kill
    mid-write leaves the old one whole. A row naming no run is appended.
    """
    run_no = row.get("run_no")
    if run_no is None:
        append_row(root, row)
        return
    line = json.dumps(strict(row), allow_nan=False)
    path = Path(root) / EPISODES_JSONL
    with _append_lock, _locked_ledger(path, "a+") as f:
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
        atomic_write_text(path, "\n".join(kept) + "\n")


def json_default(obj: Any) -> Any:
    """numpy arrays and scalars as their values; anything else as its str."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via a tempfile in the destination directory, flushed to
    disk, then ``os.replace``: a reader never sees a half-written file and
    two writers never share a tempfile."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON through :func:`atomic_write_text`."""
    atomic_write_text(path, json.dumps(payload, default=json_default, indent=2))


def write_episode_json(episode_dir: Path, row: dict) -> None:
    atomic_write_json(Path(episode_dir) / EPISODE_JSON, strict(row))


def read_episode_json(episode_dir: Path) -> dict | None:
    try:
        return json.loads((Path(episode_dir) / EPISODE_JSON).read_text())
    except (OSError, ValueError):
        return None
