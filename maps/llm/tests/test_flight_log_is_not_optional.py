"""A mission that flies is a mission that is recorded.

The flight log used to run only when ``SEARCH_LLM_FLIGHT_LOG`` named a
directory. Five callers set it; the FTE station did not, and twenty
missions flew leaving no reasoning and no tool output anywhere on disk.
Nothing failed, because a run that records nothing looks exactly like a
run with nothing to say.

So the switch is gone. These tests pin what replaced it: the destination
is DERIVED from the mission's own workdir, every caller that sets a
workdir is recorded whether it asked to be or not, and no environment
variable turns it off again.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maps.llm import http as llm_http


@pytest.fixture(autouse=True)
def _quiet_the_once_per_process_warning(monkeypatch):
    """The no-workdir warning is said once per process; reset it per test
    so each case exercises the branch rather than the first one winning."""
    monkeypatch.setattr(llm_http, "_WARNED_NO_WORKDIR", False, raising=False)


def test_a_workdir_is_all_it_takes_to_be_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    assert llm_http._resolved_flight_log_dir() == tmp_path / "flight" / "llm"


def test_the_destination_matches_what_every_caller_used_to_set_by_hand(
    monkeypatch, tmp_path
):
    """All five deleted assignments read ``<workdir>/flight/llm``. The
    derivation reproduces that path exactly, so an existing run tree and a
    new one are the same shape and every reader keeps working."""
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path / "maps_run"))
    assert llm_http._resolved_flight_log_dir() == (
        tmp_path / "maps_run" / "flight" / "llm")


def test_no_environment_variable_switches_recording_off(monkeypatch, tmp_path):
    """The vacuity floor. Setting the retired switch -- to empty, to a
    path, to anything -- must not change where or whether the mission is
    recorded, or the ablation was cosmetic."""
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    expected = tmp_path / "flight" / "llm"
    for value in ("", "0", "off", "/dev/null", str(tmp_path / "elsewhere")):
        monkeypatch.setenv("SEARCH_LLM_FLIGHT_LOG", value)
        assert llm_http._resolved_flight_log_dir() == expected

    # And nothing else in the environment reads as a disable either.
    for name in ("SEARCH_LLM_FLIGHT_LOG_DISABLE", "SEARCH_LLM_NO_FLIGHT_LOG",
                 "SEARCH_LLM_FLIGHT_LOG_OFF"):
        monkeypatch.setenv(name, "1")
    assert llm_http._resolved_flight_log_dir() == expected


def test_no_workdir_records_nothing_and_says_so(monkeypatch, caplog):
    """No workdir means no mission -- a bare transport, not a flight. That
    is the one un-recorded case, and it is audible rather than silent,
    because 'no mission is running' and 'a mission is running unrecorded'
    are the two readings of an empty transcript."""
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    with caplog.at_level("WARNING", logger=llm_http.log.name):
        assert llm_http._resolved_flight_log_dir() is None
    assert "NOTHING is recording" in caplog.text


def test_the_warning_is_said_once_not_once_per_turn(monkeypatch, caplog):
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    with caplog.at_level("WARNING", logger=llm_http.log.name):
        for _ in range(5):
            llm_http._resolved_flight_log_dir()
    assert caplog.text.count("NOTHING is recording") == 1


def test_the_repo_ships_no_caller_that_sets_the_retired_switch():
    """Derived, so it fails open -- hence the vacuity floor below: the scan
    must still find the callers that DO wire a workdir, or an empty result
    would pass this test by finding nothing at all."""
    root = Path(__file__).resolve().parents[3]
    sets_switch, sets_workdir = [], []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.sh")):
        parts = set(path.parts)
        if parts & {"node_modules", ".claude", "__pycache__", "output", "dist"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = path.relative_to(root)
        if "SEARCH_LLM_FLIGHT_LOG" in text and path != Path(__file__):
            # maps/llm/http.py names it once, in the docstring that says
            # why it is gone. Anything else is a caller re-adding it.
            if rel != Path("maps/llm/http.py"):
                sets_switch.append(str(rel))
        if "SEARCH_COMPILE_WORKDIR" in text:
            sets_workdir.append(str(rel))

    assert sets_switch == [], (
        "these files reference the retired flight-log switch; recording is "
        f"not optional and there is nothing to set: {sets_switch}")
    # Vacuity floor: the scan reached real code, and the callers that
    # actually decide where a mission records are still there.
    assert len(sets_workdir) >= 8, sets_workdir
    # The one setter every runner and the field station go through.
    pinned = "benchmarks/citynav/episode_workdir.py"
    assert pinned in sets_workdir, f"{pinned} no longer sets a workdir"


def test_the_station_and_the_bench_land_in_the_same_place(monkeypatch, tmp_path):
    """The whole point: a mission flown from the FTE seat leaves the same
    record, in the same place, as one flown from the bench."""
    seen = []
    for workdir in (tmp_path / "fte" / "calcity__14_0",
                    tmp_path / "bench" / "maps_20260824_120000_calcity_calcity__14_14"):
        monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(workdir))
        seen.append(llm_http._resolved_flight_log_dir())
    assert [p.relative_to(p.parent.parent.parent) for p in seen] == [
        Path("calcity__14_0/flight/llm"),
        Path("maps_20260824_120000_calcity_calcity__14_14/flight/llm"),
    ]


def test_the_workdir_is_expanded_like_any_other_path(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", "~/runs/ep0")
    resolved = llm_http._resolved_flight_log_dir()
    assert "~" not in str(resolved)
    assert resolved == Path(os.path.expanduser("~/runs/ep0")) / "flight" / "llm"
