"""The flight's stop path: a cancel aborts the call on the wire, and an answer
that lands as the cancel is set is never handed back or recorded."""

from __future__ import annotations

import json
import threading
import time

import pytest

from maps.llm import http
from tests.serves import ChatServe


@pytest.mark.parametrize("hold", ["before_headers", "after_headers"])
def test_a_cancel_aborts_a_call_the_serve_is_silent_on(monkeypatch, hold):
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with ChatServe(hold=hold) as serve:
        with pytest.raises(http.CallCanceled):
            http.chat_completions_raw(f"{serve.url}/v1", None, timeout=60.0, cancel_event=cancel,
                                       model="m", messages=[{"role": "user", "content": "x"}])
        assert time.monotonic() - t0 < 2.0


def test_an_answer_that_lands_as_the_cancel_is_set_is_never_used(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    cancel = threading.Event()

    def answer_as_the_stop_is_pressed(url, headers, body, *, timeout, cancel_event):
        cancel_event.set()
        return {"choices": [{"message": {"content": "late"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(http, "stream_chat_completions", answer_as_the_stop_is_pressed)
    with pytest.raises(http.CallCanceled):
        http.chat_completions_raw("http://127.0.0.1:9/v1", None, timeout=60.0, cancel_event=cancel,
                                   model="m", messages=[{"role": "user", "content": "x"}])
    kinds = [json.loads(line)["kind"]
             for line in (tmp_path / "flight" / "llm" / "transcript.jsonl").read_text().splitlines()]
    assert "response" not in kinds and "request" in kinds


def test_a_cancel_during_a_retry_backoff_ends_the_wait(monkeypatch):
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.setattr(http, "_BACKOFF_S", 30.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with ChatServe([(503, {"error": "boom"}, {})]) as serve:
        with pytest.raises(http.CallCanceled):
            http.chat_completions_raw(f"{serve.url}/v1", None, timeout=60.0, cancel_event=cancel,
                                       model="m", messages=[{"role": "user", "content": "x"}])
        assert len(serve.bodies) == 1
    assert time.monotonic() - t0 < 2.0
