"""The router-affinity key rides the env seam (maps/llm/http.py).

Unset env must mean a bit-identical request to today's: the client
carries no router fact unless the composition injects one.
"""

from maps.llm.http import _request_headers


def test_no_key_means_no_extra_header(monkeypatch):
    monkeypatch.delenv("SEARCH_LLM_ROUTING_KEY", raising=False)
    assert _request_headers(None) == {"Content-Type": "application/json"}


def test_key_rides_the_default_header(monkeypatch):
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "calcity::7#7")
    monkeypatch.delenv("SEARCH_LLM_ROUTING_KEY_HEADER", raising=False)
    assert _request_headers(None)["x-smg-routing-key"] == "calcity::7#7"


def test_header_rename_and_auth_compose(monkeypatch):
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "e1")
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY_HEADER", "x-affinity")
    headers = _request_headers("sk-x")
    assert headers["x-affinity"] == "e1"
    assert "x-smg-routing-key" not in headers
    assert headers["Authorization"] == "Bearer sk-x"


def test_empty_key_sends_nothing(monkeypatch):
    monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", "")
    assert "x-smg-routing-key" not in _request_headers(None)
