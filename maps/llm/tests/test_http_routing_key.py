"""The router-affinity key rides one header, and only when set."""

import pytest

from maps.llm.http import _request_headers


@pytest.mark.parametrize("key, want", [(None, None), ("", None), ("calcity::7#7", "calcity::7#7")])
def test_the_routing_key_header_is_sent_only_when_set(monkeypatch, key, want):
    if key is None:
        monkeypatch.delenv("SEARCH_LLM_ROUTING_KEY", raising=False)
    else:
        monkeypatch.setenv("SEARCH_LLM_ROUTING_KEY", key)
    headers = _request_headers("sk-x")
    assert headers.get("x-smg-routing-key") == want
    assert headers["Authorization"] == "Bearer sk-x" and headers["Content-Type"] == "application/json"
