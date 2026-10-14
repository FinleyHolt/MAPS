"""With an image store set, an image crosses the wire once, named by its sha;
a 409 naming a sha the store lost sends it again. The PUT carries its
Content-Type alone, a cancel aborts it, and a store fault is the store's, never an LLM status."""

from __future__ import annotations

import base64
import hashlib
import socket
import threading
import time

import httpx
import pytest

from maps.llm import http as http_mod
from tests.serves import ChatServe

_RAW = b"\x89PNG not really"
_SHA = hashlib.sha256(_RAW).hexdigest()


def _messages(url: str = "data:image/png;base64," + base64.b64encode(_RAW).decode()) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": url}},
    ]}]


def _store(monkeypatch, url: str = "http://store") -> list[tuple[str, dict]]:
    """Point the store at ``url`` with nothing held; when ``url`` is the
    default, record each PUT's (url, headers) instead of sending it."""
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.setenv(http_mod.IMAGE_STORE_ENV, url)
    monkeypatch.setattr(http_mod, "_IMAGE_STORE_HELD", set())
    puts: list[tuple[str, dict]] = []
    if url == "http://store":
        def put(self, url, **kw):
            puts.append((url, kw["headers"]))
            return httpx.Response(200, request=httpx.Request("PUT", url))
        monkeypatch.setattr(httpx.Client, "put", put)
    return puts


def _send(serve, messages=None, **kw) -> dict:
    return http_mod._chat_completions_raw(
        f"{serve.url}/v1", "sk-reasoner", model="m", messages=messages or _messages(), timeout=30.0, **kw)


def test_an_image_crosses_the_wire_once_by_its_sha(monkeypatch) -> None:
    puts = _store(monkeypatch)
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "ep-1")
    with ChatServe() as serve:
        _send(serve)
        _send(serve)
    assert [u for u, _ in puts] == [f"http://store/looks/{_SHA}"]
    assert puts[0][1] == {"Content-Type": "image/png"}
    assert all(b["messages"][0]["content"][1]["image_url"]["url"] == f"sha256:{_SHA}" for b in serve.bodies)


def test_a_sha_the_store_lost_is_sent_again(monkeypatch) -> None:
    puts = _store(monkeypatch)
    with ChatServe([(409, {"missing": [_SHA]}, {})]) as serve:
        _send(serve)
        _send(serve)
    assert [u for u, _ in puts] == [f"http://store/looks/{_SHA}"] * 2
    assert len(serve.bodies) == 3


def test_an_undecodable_image_rides_inline(monkeypatch) -> None:
    puts = _store(monkeypatch)
    bad = "data:image/png;base64,abc"
    with ChatServe() as serve:
        _send(serve, _messages(bad))
    assert puts == []
    assert serve.bodies[0]["messages"][0]["content"][1]["image_url"]["url"] == bad


def test_a_store_refusal_is_the_stores_error_and_reaches_no_reasoner(monkeypatch) -> None:
    with ChatServe() as store:  # answers a PUT 501
        _store(monkeypatch, store.url)
        with ChatServe() as serve, pytest.raises(http_mod.ImageStoreError, match="image store"):
            _send(serve)
    assert serve.bodies == []


def test_a_cancel_aborts_a_held_store_put(monkeypatch) -> None:
    with socket.create_server(("127.0.0.1", 0)) as silent:  # takes the PUT, never answers
        _store(monkeypatch, f"http://127.0.0.1:{silent.getsockname()[1]}")
        cancel = threading.Event()
        threading.Timer(0.5, cancel.set).start()
        t0 = time.monotonic()
        with ChatServe() as serve, pytest.raises(http_mod.CallCanceled):
            _send(serve, cancel_event=cancel)
        assert time.monotonic() - t0 < 2.0
    assert serve.bodies == []
