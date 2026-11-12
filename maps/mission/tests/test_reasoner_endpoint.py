"""The reasoner posts to its endpoint spec's base URL with the configured key."""

from __future__ import annotations

from dataclasses import replace

import pytest

from maps.mission import reasoner as ra
from maps.mission.tests.fakes import TEST_SPEC
from maps.settings import from_environ


@pytest.mark.parametrize("env_key, sent_key", [("sk-1", "sk-1"), (None, None), ("", None)])
def test_the_reasoner_posts_to_its_spec_with_the_env_key(monkeypatch, env_key, sent_key):
    settings = from_environ({} if env_key is None else {"SEARCH_REASONER_API_KEY": env_key})
    seen = {}

    def _fake(url, api_key, **_kw):
        seen.update(url=url, key=api_key)
        return {"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(ra, "chat_completions_raw", _fake)
    r = ra.Reasoner(
        endpoint_spec=replace(TEST_SPEC, base_url="http://spec.example/v1"), settings=settings
    )
    r.append_user("find it")
    r.step_tool_call([])
    assert seen == {"url": "http://spec.example/v1", "key": sent_key}
