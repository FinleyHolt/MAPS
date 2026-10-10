"""The launcher's filesystem contract.

``benchmarks.citynav.launch`` prints a JSON handle and writes
``.in_flight.json``; the bench driver clears that lockfile. These tests pin
the handle schema and the lockfile round-trip (dead pids read as no-run).
"""

from __future__ import annotations

import json

from benchmarks.citynav.launch import (
    _in_flight_path,
    _write_handle,
    read_in_flight,
)

HANDLE_KEYS = {
    "pid",
    "workdir",
    "log_path",
    "episode_id",
    "map",
    "split",
    "started_ts",
}


def _handle(pid: int) -> dict:
    return {
        "pid": pid,
        "workdir": "/tmp/wd",
        "log_path": "/tmp/wd/run.log",
        "episode_id": "birmingham_block_0::0#1",
        "map": "birmingham_block_0",
        "split": "val_seen",
        "started_ts": 1714839920.0,
    }


def test_in_flight_round_trip_with_live_pid(tmp_path):
    import os

    h = _handle(os.getpid())  # this test process is alive
    _write_handle(tmp_path, h)
    assert _in_flight_path(tmp_path).exists()
    got = read_in_flight(tmp_path)
    assert got is not None
    assert set(got) >= HANDLE_KEYS
    assert got == h


def test_dead_pid_reads_as_no_run(tmp_path):
    _write_handle(tmp_path, _handle(2**22 + 12345))  # certainly-dead pid
    assert read_in_flight(tmp_path) is None


def test_lockfile_is_valid_json_on_disk(tmp_path):
    import os

    _write_handle(tmp_path, _handle(os.getpid()))
    raw = json.loads(_in_flight_path(tmp_path).read_text())
    assert set(raw) >= HANDLE_KEYS
