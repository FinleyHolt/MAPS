"""Loopback fakes for the two serves a user brings: an OpenAI-compatible chat reasoner and a SAM ``/segment`` serve.

Each answers every POST to its path with the next scripted ``(status, payload, headers)`` and then its
default (any other path gets a 404), and records each request body in ``bodies``. ``hold`` makes it take the request and stay silent
for ``hold_s`` instead: ``"before_headers"`` (queued) or ``"after_headers"`` (an SSE stream that sends
nothing).
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Serve:
    path = ""
    default: tuple[int, dict, dict] = (200, {}, {})

    def __init__(self, answers=(), *, hold: str | None = None, hold_s: float = 5.0):
        self.answers = list(answers)
        self.bodies: list[dict] = []
        self.hold, self.hold_s = hold, hold_s
        self._lock = threading.Lock()
        serve = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):  # http.server's name
                serve._answer(self)

            def log_message(self, *_a):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, args=(0.05,), daemon=True).start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def _answer(self, h: BaseHTTPRequestHandler) -> None:
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length", 0))) or b"{}")
        if not h.path.endswith(self.path):
            status, payload, headers = 404, {"detail": "Not Found"}, {}
        else:
            with self._lock:
                self.bodies.append(body)
                status, payload, headers = self.answers.pop(0) if self.answers else self.default
        if self.hold:
            if self.hold == "after_headers":
                h.send_response(200)
                h.send_header("Content-Type", "text/event-stream")
                h.send_header("Transfer-Encoding", "chunked")
                h.end_headers()
                h.wfile.flush()
            time.sleep(self.hold_s)
            return
        data = json.dumps(payload).encode()
        h.send_response(status)
        for name, value in headers.items():
            h.send_header(name, value)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self._server.shutdown()
        self._server.server_close()


class ChatServe(_Serve):
    """``POST /v1/chat/completions``; unscripted calls get a one-word answer."""

    path = "/chat/completions"

    default = (200, {"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, {})


class SegmentServe(_Serve):
    """``POST /segment``; unscripted calls get a clean frame with no detections."""

    path = "/segment"

    default = (200, {"detections": []}, {})
