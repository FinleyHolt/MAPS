"""Per-step writer: what the drone detected each step.

One recorder per episode directory appends to ``detections.jsonl`` (flushed
per line, so a reader sees at most one torn final line), present from the
start: one line per detection the step's perception fold produced
(``maps.search.search_loop``'s record, masks packed) plus ``step``.
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


DETECTIONS_JSONL = "detections.jsonl"


@dataclass
class LiveRecorder:
    episode_dir: Path

    def __post_init__(self) -> None:
        self.episode_dir = Path(self.episode_dir)
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        # A mission that detects nothing has zero rows: an empty file says so.
        (self.episode_dir / DETECTIONS_JSONL).touch()

    def write_step(self, *, step: int, detections: list[dict[str, Any]]) -> None:
        for det in detections:
            _append_jsonl(self.episode_dir / DETECTIONS_JSONL, {**det, "step": int(step)})
