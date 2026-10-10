"""Wall-clock timeouts are counted first-class, and stay in the denominator.

A ``timed_out`` row marks the episode wall-clock cap binding (the bound
that matters under an unbounded step budget). It is an ALGORITHM failure:
counted separately in ``ReadStats.timed_out`` so the report can show it
beside SR, but never dropped like an infra abort.
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.citynav.analysis.rows import read_attempts


def _write_ledger(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_timed_out_rows_are_counted_and_kept(tmp_path: Path):
    ledger = _write_ledger(
        tmp_path / "episodes.jsonl",
        [
            # Clean scored row.
            {"episode_id": "calcity::1#0", "status": "ok",
             "metrics": {"SR": 1, "NE": 3.0}},
            # Timeout with no committed record: a no-metrics miss.
            {"episode_id": "calcity::2#1", "status": "timed_out"},
            # Timeout that still committed: scored AND counted as timeout.
            {"episode_id": "calcity::3#2", "status": "timed_out",
             "metrics": {"SR": 0, "NE": 900.0}},
            # Infra abort: dropped from attempts, not a timeout.
            {"episode_id": "calcity::4#3", "status": "infra",
             "error": "killed_midrun"},
        ],
    )

    attempts, stats = read_attempts([ledger])

    assert stats.timed_out == 2
    assert stats.rows == 4
    assert stats.infra == 1
    # Both timeout rows stay attempts (the denominator); the infra row is
    # dropped.
    kept = {a.episode_id for a in attempts}
    assert kept == {"calcity::1#0", "calcity::2#1", "calcity::3#2"}
