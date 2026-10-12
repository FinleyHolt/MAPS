#!/usr/bin/env python
"""The mission's reasoning effort reaches the WIRE, on every request."""
from __future__ import annotations

import json

import pytest

from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import MissionReply, run_mission
from maps.search.tests.fakes import TEST_SPEC, bare_ctx, unarmed_boost


def _serve(monkeypatch) -> list[dict]:
    """Pin the transport to a plain text turn; return every sent body."""
    sent: list[dict] = []

    def _fake(url, api_key, **kwargs):
        sent.append(kwargs)
        return {"choices": [{
            "finish_reason": "stop",
            "message": {"content": "standing by", "reasoning_content": "",
                        "tool_calls": []},
        }]}

    monkeypatch.setattr(reasoner_agent, "_chat_completions_raw", _fake)
    return sent


@pytest.mark.parametrize("word", ["0.3", "0.0"])
def test_the_missions_word_rides_every_request_and_the_arms_does_not(monkeypatch, word):
    """Two rounds on one reasoner both carry the mission's word as a string; "0.0" is a
    value (thinking off), not an absence."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC,
                 reasoning_effort=word)
    r.append_user("find the truck")
    r.step_tool_call([])
    r.append_user("and the one north of it")
    r.step_tool_call([])
    assert [b["reasoning_effort"] for b in sent] == [word, word]


def test_a_reasoner_with_no_word_sends_the_arms(monkeypatch):
    """VACUITY FLOOR: the environment is still the default, read at CALL time, so a rig
    select mid-day reaches a mission that named none."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.9")
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC)
    r.append_user("find the truck")
    r.step_tool_call([])
    assert sent[0]["reasoning_effort"] == "0.9"


def test_the_effort_reaches_no_served_byte(monkeypatch):
    """The model is never told a budget fact."""
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC,
                 reasoning_effort="0.3")
    r.append_user("find the truck")
    r.step_tool_call([{"type": "function", "function": {"name": "run"}}])
    served = json.dumps([sent[0]["messages"], sent[0]["tools"]]).lower()
    assert "effort" not in served and "0.3" not in served


def test_run_mission_hands_the_word_to_the_reasoner_it_builds(monkeypatch):
    """THE mission entry owns the Reasoner, so the station's word has to survive that
    construction; nothing downstream can put it back."""
    seen: dict = {}

    def _round(*, reasoner, **kw):
        seen["reasoner"] = reasoner
        return MissionReply(ended_by="reply")

    monkeypatch.setattr(
        "maps.search.parse_compile.reasoner_loop.advance_reasoner_one_round",
        _round)
    _reply, reasoner = run_mission(
        instruction="find the truck",
        reasoner_url="http://unused",
        endpoint_spec=TEST_SPEC,
        ctx=bare_ctx(),
        mission_spec=None,
        boost_queue=unarmed_boost,
        reasoning_effort="0.6",
    )
    assert seen["reasoner"] is reasoner
    assert reasoner.reasoning_effort == "0.6"
