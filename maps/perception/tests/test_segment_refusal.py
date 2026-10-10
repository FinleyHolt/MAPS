"""A ``/segment`` request the segmenter refuses is the client's own fault, not an outage.

A 4xx other than the serve's weather (408 request timeout, 429 rate limit) is
the request this client built, refused: every replica refuses it the same way.
It raises ``SamRequestRefused`` at once, which is not a ``SamServerOutage``.
A 408 or 429 moves the call to another replica like a 5xx.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import maps.perception.sam.detect_client as dc


@pytest.fixture
def replicas(monkeypatch):
    """Two local ``/segment`` replicas answering scripted statuses in turn
    (then 200); yields ``(urls, script, seen)``."""
    script: dict[int, list[int]] = {0: [], 1: []}
    seen: list[tuple[int, int]] = []
    servers = []
    for idx in (0, 1):
        class Handler(BaseHTTPRequestHandler):
            replica = idx

            def do_POST(self):  # http.server's name
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                queue = script[self.replica]
                status = queue.pop(0) if queue else 200
                seen.append((self.replica, status))
                body = json.dumps({"detail": "concepts: too many items"} if status != 200
                                  else {"image_shape_hw": [1008, 1008], "detections": []}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_a):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    monkeypatch.setattr(dc, "time", SimpleNamespace(sleep=lambda _s: None, perf_counter=time.perf_counter))
    yield [f"http://127.0.0.1:{s.server_address[1]}" for s in servers], script, seen
    for s in servers:
        s.shutdown()


def _picker(urls):
    turn = iter(range(100))
    return lambda: urls[next(turn) % len(urls)]


def test_a_refused_request_raises_at_once_and_is_not_an_outage(replicas):
    urls, script, seen = replicas
    script[0] = [422]
    with pytest.raises(dc.SamRequestRefused, match="422") as caught:
        dc._post_segment_avoid_stale_pool(picker=_picker(urls), json_body={"image_b64": ""})
    assert seen == [(0, 422)]
    assert not isinstance(caught.value, dc.SamServerOutage)


@pytest.mark.parametrize("status", [408, 429])
def test_the_serves_weather_moves_the_call_to_another_replica(replicas, status):
    urls, script, seen = replicas
    script[0] = [status]
    payload = dc._post_segment_avoid_stale_pool(picker=_picker(urls), json_body={"image_b64": ""})
    assert payload == {"image_shape_hw": [1008, 1008], "detections": []}
    assert seen == [(0, status), (1, 200)]
