"""The flight's stop path: a cancel aborts the call on the wire, and an answer
that lands as the cancel is set is never handed back, booked or recorded."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from maps.llm import http

_SILENT_S = 5.0


class _SilentServe(BaseHTTPRequestHandler):
    """Takes the request, then says nothing for ``_SILENT_S``: queued (no
    headers yet) or generating (headers sent, no chunk yet)."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.server.mode == "headers_then_silent":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
        time.sleep(_SILENT_S)

    def log_message(self, *_a):
        pass


@pytest.mark.parametrize("mode", ["silent_before_headers", "headers_then_silent"])
def test_a_cancel_aborts_a_call_the_serve_is_silent_on(monkeypatch, mode):
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SilentServe)
    srv.mode = mode
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = time.monotonic()
    try:
        with pytest.raises(http.CallCanceled):
            http._chat_completions_raw(
                f"http://127.0.0.1:{srv.server_port}/v1", None, timeout=60.0,
                cancel_event=cancel, model="m", messages=[{"role": "user", "content": "x"}])
        assert time.monotonic() - t0 < 2.0
    finally:
        srv.shutdown()
        srv.server_close()


def test_an_answer_that_lands_as_the_cancel_is_set_is_never_used(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_COMPILE_WORKDIR", str(tmp_path))
    monkeypatch.setenv("NET_METER_LOG", str(tmp_path / "net_meter.jsonl"))
    cancel = threading.Event()

    def answer_as_the_stop_is_pressed(url, headers, body, *, timeout, cancel_event):
        cancel_event.set()
        return {"choices": [{"message": {"content": "late"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1}}

    monkeypatch.setattr(http, "_stream_chat_completions", answer_as_the_stop_is_pressed)
    with pytest.raises(http.CallCanceled):
        http._chat_completions_raw("http://127.0.0.1:9/v1", None, timeout=60.0, cancel_event=cancel,
                                   model="m", messages=[{"role": "user", "content": "x"}])
    kinds = [json.loads(line)["kind"]
             for line in (tmp_path / "flight" / "llm" / "transcript.jsonl").read_text().splitlines()]
    assert "response" not in kinds and "request" in kinds
    assert not (tmp_path / "net_meter.jsonl").exists()
