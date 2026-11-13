"""The sortie ledger ``upsert_row`` rewrites: one row per run, never seen empty or cut."""

from __future__ import annotations

import json
import threading

import pytest

from benchmarks.run_record import episodes


def _runs(root) -> list[int]:
    text = (root / episodes.EPISODES_JSONL).read_text()
    return [json.loads(line)["run_no"] for line in text.splitlines() if line]


def test_upsert_replaces_its_run_and_appends_a_new_one(tmp_path) -> None:
    for run_no, n in ((1, 0), (2, 0), (1, 1)):
        episodes.upsert_row(tmp_path, {"run_no": run_no, "n": n})
    rows = [json.loads(x) for x in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert rows == [{"run_no": 1, "n": 1}, {"run_no": 2, "n": 0}]


def test_a_reader_never_sees_the_ledger_short_of_its_rows(tmp_path) -> None:
    for run_no in range(1, 41):
        episodes.upsert_row(tmp_path, {"run_no": run_no, "pad": "x" * 200})
    seen: list[int] = []
    done = threading.Event()

    def read() -> None:
        while not done.is_set():
            seen.append(len(_runs(tmp_path)))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for i in range(400):
            episodes.upsert_row(tmp_path, {"run_no": 1 + i % 40, "pad": "y" * 200})
    finally:
        done.set()
        reader.join()
    assert seen and min(seen) == 40


def test_a_write_that_fails_leaves_the_ledger_it_found(tmp_path, monkeypatch) -> None:
    episodes.upsert_row(tmp_path, {"run_no": 1, "n": 0})
    before = (tmp_path / "episodes.jsonl").read_bytes()

    def fail(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(episodes.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        episodes.upsert_row(tmp_path, {"run_no": 1, "n": 1})
    assert (tmp_path / "episodes.jsonl").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["episodes.jsonl"]


def test_a_rewritten_ledger_keeps_its_mode(tmp_path):
    """The rewrite lands through a tempfile; the ledger keeps the mode it had,
    and a ledger the upsert starts gets the mode an append gives."""
    import os
    import stat

    appended, upserted = tmp_path / "a", tmp_path / "u"
    appended.mkdir()
    upserted.mkdir()
    episodes.append_row(appended, {"episode_id": "e", "run_no": 1})
    episodes.upsert_row(upserted, {"episode_id": "e", "run_no": 1})
    mode = lambda root: stat.S_IMODE((root / episodes.EPISODES_JSONL).stat().st_mode)
    assert mode(upserted) == mode(appended)

    os.chmod(upserted / episodes.EPISODES_JSONL, 0o640)
    episodes.upsert_row(upserted, {"episode_id": "e", "run_no": 1, "status": "ok"})
    assert mode(upserted) == 0o640
