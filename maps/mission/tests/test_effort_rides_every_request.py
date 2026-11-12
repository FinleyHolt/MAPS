#!/usr/bin/env python
"""The mission's reasoning effort reaches the WIRE, on every request."""

from __future__ import annotations

import json

import pytest

from maps.mission import reasoner
from maps.settings import from_environ
from maps.mission.reasoner import Reasoner
from maps.mission.spec import MissionReply
from maps.mission.loop import run_mission
from maps.mission.tests.fakes import TEST_SPEC, bare_ctx, unarmed_boost


def _serve(monkeypatch) -> list[dict]:
    """Pin the transport to a plain text turn; return every sent body."""
    sent: list[dict] = []

    def _fake(url, api_key, *, workdir=None, routing_key=None, endpoint=None, **kwargs):
        sent.append(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "standing by",
                        "reasoning_content": "",
                        "tool_calls": [],
                    },
                }
            ]
        }

    monkeypatch.setattr(reasoner, "chat_completions_raw", _fake)
    return sent


@pytest.mark.parametrize("word", ["0.3", "0.0"])
def test_the_missions_word_rides_every_request_and_the_arms_does_not(monkeypatch, word):
    """Two rounds on one reasoner both carry the mission's word as a string; "0.0" is a
    value (thinking off), not an absence."""
    sent = _serve(monkeypatch)
    r = Reasoner(
        endpoint_spec=TEST_SPEC,
        settings=from_environ({"SEARCH_SAMPLING_REASONING_EFFORT": "0.99"}),
        reasoning_effort=word,
    )
    r.append_user("find the truck")
    r.step_tool_call([])
    r.append_user("and the one north of it")
    r.step_tool_call([])
    assert [b["reasoning_effort"] for b in sent] == [word, word]


def test_a_reasoner_with_no_word_sends_the_arms(monkeypatch):
    """VACUITY FLOOR: the arm's effort (the run's settings) is still the default for a
    mission that named none."""
    sent = _serve(monkeypatch)
    r = Reasoner(
        endpoint_spec=TEST_SPEC, settings=from_environ({"SEARCH_SAMPLING_REASONING_EFFORT": "0.9"})
    )
    r.append_user("find the truck")
    r.step_tool_call([])
    assert sent[0]["reasoning_effort"] == "0.9"


def test_the_effort_reaches_no_served_byte(monkeypatch):
    """The model is never told a budget fact."""
    sent = _serve(monkeypatch)
    r = Reasoner(endpoint_spec=TEST_SPEC, reasoning_effort="0.3")
    r.append_user("find the truck")
    r.step_tool_call([{"type": "function", "function": {"name": "run"}}])
    served = json.dumps([sent[0]["messages"], sent[0]["tools"]]).lower()
    assert "effort" not in served and "0.3" not in served


def test_run_mission_hands_the_word_to_the_reasoner_it_builds(monkeypatch):
    """THE mission entry owns the Reasoner, so the flight mission's word has to survive that
    construction; nothing downstream can put it back."""
    seen: dict = {}

    def _converse(mission, user_message):
        seen["reasoner"] = mission.reasoner
        return MissionReply(ended_by="reply")

    monkeypatch.setattr("maps.mission.loop._Mission.converse", _converse)
    _reply, mission = run_mission(
        instruction="find the truck",
        endpoint_spec=TEST_SPEC,
        ctx=bare_ctx(),
        mission_spec=None,
        boost_queue=unarmed_boost,
        reasoning_effort="0.6",
    )
    assert seen["reasoner"] is mission.reasoner
    assert mission.reasoner.reasoning_effort == "0.6"
