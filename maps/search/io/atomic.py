"""The one atomic JSON write.

Every on-disk artifact a run produces while the dashboard may be polling it
goes through here, so a reader never sees a half-written file: write to a
tempfile in the destination directory, then ``os.replace`` (atomic within a
filesystem). ``sort_keys`` gives checkpoint writers a stable diff across runs.

Lives inside ``maps/search/`` to keep the package robot-agnostic and free
of benchmark-side imports.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    """Make numpy and Path JSON-serialisable; stringify anything else."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def atomic_write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Write ``payload`` as JSON to ``path`` via tmp + ``os.replace``.

    The tempfile is minted with ``mkstemp`` in the destination directory --
    not a fixed ``.tmp`` suffix -- so two writers racing on one path cannot
    clobber each other's partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=_json_default, indent=2,
                      sort_keys=sort_keys)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
