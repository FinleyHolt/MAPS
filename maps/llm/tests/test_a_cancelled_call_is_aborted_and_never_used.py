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
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with ChatServe(hold=hold) as serve:
        with pytest.raises(http.CallCanceled):
            http.chat_completions_raw(
                f"{serve.url}/v1",
                None,
                timeout=60.0,
                cancel_event=cancel,
                model="m",
                messages=[{"role": "user", "content": "x"}],
            )
        assert time.monotonic() - t0 < 2.0


def test_an_answer_that_lands_as_the_cancel_is_set_is_never_used(monkeypatch, tmp_path):
    cancel = threading.Event()

    def answer_as_the_stop_is_pressed(url, headers, body, *, timeout, cancel_event):
        cancel_event.set()
        return {
            "choices": [{"message": {"content": "late"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

    monkeypatch.setattr(http, "stream_chat_completions", answer_as_the_stop_is_pressed)
    with pytest.raises(http.CallCanceled):
        http.chat_completions_raw(
            "http://127.0.0.1:9/v1",
            None,
            timeout=60.0,
            cancel_event=cancel,
            workdir=tmp_path,
            model="m",
            messages=[{"role": "user", "content": "x"}],
        )
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "flight" / "llm" / "transcript.jsonl").read_text().splitlines()
    ]
    assert "response" not in kinds and "request" in kinds


def test_a_cancel_during_a_retry_backoff_ends_the_wait(monkeypatch):
    monkeypatch.setattr(http, "_BACKOFF_S", 30.0)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    with ChatServe([(503, {"error": "boom"}, {})]) as serve:
        with pytest.raises(http.CallCanceled):
            http.chat_completions_raw(
                f"{serve.url}/v1",
                None,
                timeout=60.0,
                cancel_event=cancel,
                model="m",
                messages=[{"role": "user", "content": "x"}],
            )
        assert len(serve.bodies) == 1
    assert time.monotonic() - t0 < 2.0


def _chat(port, cancel):
    http.chat_completions_raw(
        f"http://127.0.0.1:{port}/v1",
        None,
        timeout=60.0,
        cancel_event=cancel,
        model="m",
        messages=[{"role": "user", "content": "x"}],
    )


def _put(port, cancel):
    http._put_image(f"http://127.0.0.1:{port}/images/x", b"x", "image/jpeg", cancel)


@pytest.mark.parametrize("send", [_chat, _put])
def test_a_cancel_during_a_connect_that_never_completes_ends_the_call(monkeypatch, send):
    """A listener whose accept queue is full drops every further SYN, so the
    connect hangs as on a blackholed route; the connect bound ends it and the
    cancel lands before any retry."""
    import socket

    monkeypatch.setattr(http, "_CONNECT_TIMEOUT_S", 0.5)
    lis = socket.socket()
    lis.bind(("127.0.0.1", 0))
    lis.listen(0)
    fillers = []
    while True:  # fill the accept queue until a connect hangs
        s = socket.socket()
        s.settimeout(0.3)
        fillers.append(s)
        try:
            s.connect(lis.getsockname())
        except OSError:
            break
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    got = []

    def call():
        try:
            send(lis.getsockname()[1], cancel)
        except Exception as e:
            got.append(e)

    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join(10.0)
    for s in fillers + [lis]:
        s.close()
    assert not t.is_alive(), "the connect outlived its bound"
    assert len(got) == 1 and isinstance(got[0], http.CallCanceled), got
