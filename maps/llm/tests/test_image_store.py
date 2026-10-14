"""With an image store set, an image crosses the wire once, named by its sha;
a 409 naming a sha the store lost sends it again."""

from __future__ import annotations

import base64
import hashlib

from maps.llm import http as http_mod
from tests.serves import ChatServe

_RAW = b"\x89PNG not really"
_SHA = hashlib.sha256(_RAW).hexdigest()
_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "look"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(_RAW).decode()}},
]}]


def _send(monkeypatch, answers) -> tuple[list[dict], list[str]]:
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.setenv(http_mod.IMAGE_STORE_ENV, "http://store")
    monkeypatch.setattr(http_mod, "_IMAGE_STORE_HELD", set())
    puts: list[str] = []

    class _Ok:
        def raise_for_status(self) -> None:
            pass

    monkeypatch.setattr(http_mod.httpx, "put", lambda url, **kw: (puts.append(url), _Ok())[1])
    with ChatServe(answers) as serve:
        http_mod._chat_completions_raw(f"{serve.url}/v1", None, model="m", messages=_MESSAGES, timeout=30.0)
        http_mod._chat_completions_raw(f"{serve.url}/v1", None, model="m", messages=_MESSAGES, timeout=30.0)
    return serve.bodies, puts


def test_an_image_crosses_the_wire_once_by_its_sha(monkeypatch) -> None:
    bodies, puts = _send(monkeypatch, [])
    assert puts == [f"http://store/looks/{_SHA}"]
    assert all(b["messages"][0]["content"][1]["image_url"]["url"] == f"sha256:{_SHA}" for b in bodies)
    assert _MESSAGES[0]["content"][1]["image_url"]["url"].startswith("data:")


def test_a_sha_the_store_lost_is_sent_again(monkeypatch) -> None:
    bodies, puts = _send(monkeypatch, [(409, {"missing": [_SHA]}, {})])
    assert puts == [f"http://store/looks/{_SHA}"] * 2
    assert len(bodies) == 3


def test_an_undecodable_image_rides_inline(monkeypatch) -> None:
    monkeypatch.delenv("SEARCH_COMPILE_WORKDIR", raising=False)
    monkeypatch.setenv(http_mod.IMAGE_STORE_ENV, "http://store")
    monkeypatch.setattr(http_mod, "_IMAGE_STORE_HELD", set())
    puts: list[str] = []
    monkeypatch.setattr(http_mod.httpx, "put", lambda url, **kw: puts.append(url))
    bad = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}]
    with ChatServe() as serve:
        http_mod._chat_completions_raw(f"{serve.url}/v1", None, model="m", messages=bad, timeout=30.0)
    assert puts == []
    assert serve.bodies[0]["messages"] == bad
