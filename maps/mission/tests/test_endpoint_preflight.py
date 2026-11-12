"""The preflight flies its two turns against a loopback fake of a hosted
Responses endpoint, and fails an effort whose second turn reads no prompt-cache hit."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from maps.mission import endpoint_preflight
from maps.llm import endpoint_spec

ENCRYPTED = "gAAAAB-opaque-reasoning-turn-1"


def _fake_endpoint(cached_on_second: int | None):
    """A loopback server answering ``/models`` and ``/responses`` as Azure's v1
    surface does: the first turn blocking, the second streamed."""
    bodies: list[dict] = []
    paths: list[str] = []
    gets: list[str] = []

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
            gets.append(self.path)
            self.server.gets.append(self.path)
            self._send(json.dumps({"data": [{"id": "gpt-5.6-sol-2026-07-09"}]}).encode())

        def do_POST(self):
            paths.append(self.path)
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            n = len(bodies)
            usage = {"input_tokens": 3000, "output_tokens": 40, "total_tokens": 3040,
                     "output_tokens_details": {"reasoning_tokens": 25},
                     **({} if cached_on_second is None else
                        {"input_tokens_details": {"cached_tokens": 0 if n == 1 else cached_on_second}})}
            response = {"id": f"resp_{n}", "model": "gpt-5.6-sol-2026-07-09", "status": "completed",
                        "usage": usage, "output": [
                            {"type": "reasoning", "id": f"rs_{n}", "summary": [], "encrypted_content": ENCRYPTED},
                            {"type": "message", "role": "assistant",
                             "content": [{"type": "output_text", "text": "red"}]},
                            {"type": "function_call", "call_id": f"c{n}", "name": "run",
                             "arguments": "{\"code\": \"print('ok')\"}"}]}
            if not body.get("stream"):
                self._send(json.dumps(response).encode())
                return
            events = [{"type": "response.created", "response": {"id": f"resp_{n}"}},
                      {"type": "response.output_text.delta", "delta": "red"},
                      {"type": "response.completed", "response": response}]
            self._send("".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode(),
                       "text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.gets = gets
    server.gets = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, bodies, paths


@pytest.fixture
def hosted(monkeypatch):
    def start(cached_on_second):
        server, bodies, paths = _fake_endpoint(cached_on_second)
        monkeypatch.setenv("SEARCH_REASONER_URL", f"http://127.0.0.1:{server.server_port}/openai/v1")
        monkeypatch.setenv("SEARCH_ENDPOINT_SPEC_OVERRIDE", "gpt-5.6-sol:262144")
        monkeypatch.setenv("SEARCH_REASONER_API_KEY", "fake-key")
        monkeypatch.delenv("SEARCH_REASONER_MODELS_URL", raising=False)
        monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
        monkeypatch.delenv("SEARCH_ENDPOINT_OFFLINE", raising=False)
        # Discovery's tokenizer load is the hub's; these tests own the endpoint only.
        monkeypatch.setattr(endpoint_spec, "_require_tokenizer", lambda m, override="": f"tok:{m}")
        return server, bodies, paths
    servers = []
    yield lambda c: servers.append(start(c)) or servers[-1]
    for server, *_ in servers:
        server.shutdown()


def test_a_second_turn_that_reads_the_cache_passes(hosted, capsys) -> None:
    _, bodies, paths = hosted(2944)
    assert endpoint_preflight.main(["--effort", "xhigh"]) == 0
    out = capsys.readouterr().out
    assert '"second_turn_cached_tokens": 2944' in out and "gpt-5.6-sol-2026-07-09" in out
    assert paths == ["/openai/v1/responses"] * 2 and bodies[1]["stream"] is True
    keys = {b["prompt_cache_key"] for b in bodies}
    assert len(keys) == 1 and keys != {None}, "both turns ride one cache key"
    assert all(b["reasoning"] == {"effort": "xhigh"} and b["store"] is False
               and b["include"] == ["reasoning.encrypted_content"] for b in bodies)
    # The first turn's encrypted reasoning rides back ahead of its call on the second.
    kinds = [(i.get("type"), i.get("encrypted_content")) for i in bodies[1]["input"]]
    at = kinds.index(("reasoning", ENCRYPTED))
    assert kinds[at + 1][0] in ("message", "function_call") and ("function_call_output", None) in kinds[at:]


@pytest.mark.parametrize("cached", [0, None], ids=["zero-hit", "no-meter"])
def test_a_second_turn_that_misses_the_cache_fails_the_effort(hosted, capsys, cached) -> None:
    hosted(cached)
    assert endpoint_preflight.main(["--effort", "medium"]) == 1
    assert "NoCacheHit" in capsys.readouterr().err


def test_an_offline_preflight_asks_no_model_listing_and_still_flies_both_turns(hosted, monkeypatch) -> None:
    server, bodies, _ = hosted(2944)
    monkeypatch.setenv("SEARCH_ENDPOINT_OFFLINE", "1")
    assert endpoint_preflight.main(["--effort", "xhigh"]) == 0
    assert server.gets == [] and len(bodies) == 2
    monkeypatch.delenv("SEARCH_ENDPOINT_OFFLINE")
    assert endpoint_preflight.main(["--effort", "xhigh"]) == 0
    assert server.gets == ["/openai/v1/models"]
