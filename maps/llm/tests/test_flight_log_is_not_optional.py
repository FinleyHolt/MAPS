"""The flight log lands under the mission's workdir; with no workdir nothing records, and the log says so once."""

from __future__ import annotations

import os
from pathlib import Path

from maps.llm import http as llm_http


def test_no_workdir_records_nothing_and_says_so_once(monkeypatch, caplog):
    monkeypatch.setattr(llm_http, "_WARNED_NO_WORKDIR", False)
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    with caplog.at_level("WARNING", logger=llm_http.log.name):
        for _ in range(5):
            assert llm_http._resolved_flight_log_dir() is None
    assert caplog.text.count("NOTHING is recording") == 1


def test_the_workdir_is_expanded_like_any_other_path(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", "~/runs/ep0")
    assert llm_http._resolved_flight_log_dir() == Path(os.path.expanduser("~/runs/ep0")) / "flight" / "llm"
