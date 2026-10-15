#!/usr/bin/env python
"""A collapsed completion is a SERVE fault, not the model's empty turn."""
from __future__ import annotations

import pytest

from maps.llm.http import DegenerateCompletion
from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.tests.fakes import TEST_SPEC


def _serve(monkeypatch, *, content="", reasoning="", finish_reason="stop",
           tool_calls=()):
    """Pin the transport to one canned completion; return the sent kwargs."""
    sent: dict = {}

    def _fake(url, api_key, **kwargs):
        sent.update(kwargs)
        return {"choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content, "reasoning_content": reasoning,
                        "tool_calls": list(tool_calls)},
        }]}

    monkeypatch.setattr(reasoner_agent, "_chat_completions_raw", _fake)
    return sent


def _reasoner() -> Reasoner:
    return Reasoner(endpoint_spec=TEST_SPEC)


@pytest.mark.parametrize("channel", ["reasoning", "content"])
def test_a_collapse_in_either_channel_raises_and_leaves_history_clean(monkeypatch, channel):
    """A few clean chars then token 0 forever, in the think span or after it."""
    _serve(monkeypatch, **{channel: "These" + "!" * 400}, finish_reason="length")
    r = _reasoner()
    before = len(r.messages)
    with pytest.raises(DegenerateCompletion):
        r.step_tool_call([])
    assert len(r.messages) == before


def test_length_clip_with_real_reasoning_is_NOT_degenerate(monkeypatch):
    """A think span that ran long is a CLIP, and clips keep their repair."""
    _serve(monkeypatch,
           reasoning="I need to weigh the two candidate rooftops. " * 40,
           finish_reason="length")
    r = _reasoner()
    before = len(r.messages)
    out = r.step_tool_call([])
    assert out["kind"] == "text"
    assert len(r.messages) == before + 1


def test_empty_turn_still_reads_as_a_text_turn(monkeypatch):
    """VACUITY FLOOR: the model's own empty turn is a DIFFERENT thing."""
    _serve(monkeypatch, reasoning="I have nothing further to add.")
    r = _reasoner()
    before = len(r.messages)
    out = r.step_tool_call([])
    assert out["kind"] == "text"
    assert out["content"] == ""
    assert len(r.messages) == before + 1
