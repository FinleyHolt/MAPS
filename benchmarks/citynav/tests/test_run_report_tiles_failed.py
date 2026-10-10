"""run_report says when SAHI tiles failed on the scored rows.

``metrics.sam_tiles_failed`` counts tiles the segmenter could not answer;
their objects are missing from those frames, so a run that lost any says so
beside its SR.
"""

from __future__ import annotations

import json

from benchmarks.citynav.analysis import run_report


def _row(eid: str, failed: int) -> dict:
    return {"episode_id": eid, "worker_id": 0, "status": "ok", "elapsed_s": 1.0,
            "workdir": f"w0/maps_20260101T000000Z_citynav_{eid}",
            "metrics": {"episode_id": eid, "SR": 0, "NE": 10.0, "sam_tiles_failed": failed}}


def test_failed_tiles_are_reported_beside_the_sr(tmp_path, capsys):
    rows = [_row("a", 3), _row("b", 0), _row("c", 2)]
    (tmp_path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    run_report.report(tmp_path)
    out = capsys.readouterr().out
    assert "SAHI tiles failed = 5 over 2 of 3 scored episode(s)" in out, out


def test_a_run_that_lost_no_tile_prints_no_line(tmp_path, capsys):
    (tmp_path / "episodes.jsonl").write_text(json.dumps(_row("a", 0)) + "\n")
    run_report.report(tmp_path)
    assert "SAHI tiles failed" not in capsys.readouterr().out
