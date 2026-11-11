"""Ledger rows as the bench driver writes them, for the reader tests."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run_record.rows import EPISODES_JSONL


def ledger_row(eid: str, stamp: str, sr: int | None = None, *, status: str = "ok",
               worker: str = "w0", **extra) -> dict:
    """One attempt of ``eid`` started at ``stamp``; ``extra`` keys replace the defaults."""
    slug = eid.replace("::", "__").replace("#", "_")
    row: dict = {
        "episode_id": eid, "worker_id": worker, "status": status, "error": None,
        "elapsed_s": 1.0, "workdir": f"{worker}/maps_{stamp}_citynav_{slug}",
    }
    if sr is not None:
        row["metrics"] = {"episode_id": eid, "SR": sr, "OSR": sr, "NE": 10.0}
    row.update(extra)
    return row


def write_ledger(root: Path, *rows: dict, name: str = EPISODES_JSONL) -> Path:
    """Write ``rows`` one JSON per line to ``root/name`` and return ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root
