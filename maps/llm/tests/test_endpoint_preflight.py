"""The preflight flies its two turns against a loopback fake of a hosted
endpoint, and fails an effort whose second turn reads no prompt-cache hit."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from maps.llm import endpoint_preflight


def _fake_endpoint(cached_on_second: int | None):
    """A loopback server answering ``/models`` and ``/chat/completions`` as Azure's
    v1 surface does: the first turn blocking, the second streamed."""
    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, payload: bytes, ctype: str = "application/json") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._send(json.dumps({"data": [{"id": "gpt-5.6-sol-2026-07-09"}]}).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            details = {"cached_tokens": 0 if len(bodies) == 1 else cached_on_second}
            usage = {"prompt_tokens": 3000, "completion_tokens": 40, "total_tokens": 3040,
                     **({} if cached_on_second is None else {"prompt_tokens_details": details})}
            call = {"id": f"c{len(bodies)}", "type": "function",
                    "function": {"name": "run", "arguments": "{\"code\": \"print('ok')\"}"}}
            if not body.get("stream"):
                self._send(json.dumps({"model": "gpt-5.6-sol-2026-07-09", "usage": usage, "choices": [
                    {"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "red",
                                                                "tool_calls": [call]}}]}).encode())
                return
            events = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
                      {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]},
                                    "finish_reason": "tool_calls"}]},
                      {"choices": [], "model": "gpt-5.6-sol-2026-07-09", "usage": usage}]
            self._send("".join(f"data: {json.dumps(e)}\n\n" for e in events).encode()
                       + b"data: [DONE]\n\n", "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, bodies


@pytest.fixture
def hosted(monkeypatch):
    def start(cached_on_second):
        server, bodies = _fake_endpoint(cached_on_second)
        monkeypatch.setenv("SEARCH_REASONER_URL", f"http://127.0.0.1:{server.server_port}/openai/v1")
        monkeypatch.setenv("SEARCH_ENDPOINT_SPEC_OVERRIDE", "gpt-5.6-sol:262144")
        monkeypatch.setenv("SEARCH_REASONER_API_KEY", "fake-key")
        monkeypatch.delenv("SEARCH_REASONER_MODELS_URL", raising=False)
        monkeypatch.delenv("SEARCH_LLM_ROUTING_KEY", raising=False)
        monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
        return server, bodies
    servers = []
    yield lambda c: servers.append(start(c)) or servers[-1]
    for server, _ in servers:
        server.shutdown()


def test_a_second_turn_that_reads_the_cache_passes(hosted, capsys) -> None:
    _, bodies = hosted(2944)
    assert endpoint_preflight.main(["--effort", "xhigh"]) == 0
    out = capsys.readouterr().out
    assert '"second_turn_cached_tokens": 2944' in out and "gpt-5.6-sol-2026-07-09" in out
    assert len(bodies) == 2 and bodies[1]["stream"] is True
    keys = {b["prompt_cache_key"] for b in bodies}
    assert len(keys) == 1 and keys != {None}, "both turns ride one cache key"
    assert all(b["reasoning_effort"] == "xhigh" for b in bodies)


@pytest.mark.parametrize("cached", [0, None], ids=["zero-hit", "no-meter"])
def test_a_second_turn_that_misses_the_cache_fails_the_effort(hosted, capsys, cached) -> None:
    hosted(cached)
    assert endpoint_preflight.main(["--effort", "medium"]) == 1
    assert "NoCacheHit" in capsys.readouterr().err
