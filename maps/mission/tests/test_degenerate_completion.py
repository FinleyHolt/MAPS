#!/usr/bin/env python
"""A collapsed completion is a SERVE fault, not the model's empty turn."""

from __future__ import annotations

import pytest

from maps.mission import reasoner
from maps.mission.reasoner import DegenerateCompletion, Reasoner
from maps.mission.tests.fakes import TEST_SPEC


def _serve(monkeypatch, *, content="", reasoning="", finish_reason="stop", tool_calls=(), items=()):
    """Pin the transport to one canned completion; return the sent kwargs."""
    sent: dict = {}

    def _fake(url, api_key, *, workdir=None, routing_key=None, endpoint=None, **kwargs):
        sent.update(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": content,
                        "reasoning_content": reasoning,
                        "tool_calls": list(tool_calls),
                        **({"reasoning_items": list(items)} if items else {}),
                    },
                }
            ]
        }

    monkeypatch.setattr(reasoner, "chat_completions_raw", _fake)
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
    _serve(
        monkeypatch,
        reasoning="I need to weigh the two candidate rooftops. " * 40,
        finish_reason="length",
    )
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


def test_a_tool_call_the_output_cap_cut_off_is_no_call(monkeypatch):
    """A cut call never runs: the turn reads as a truncated text turn, and
    the history carries no call id waiting on a result."""
    call = {"id": "c1", "type": "function"}
    call["function"] = {"name": "run", "arguments": '{"code": "go_to(prior_0'}
    _serve(monkeypatch, reasoning="settled", finish_reason="length", tool_calls=[call])
    r = _reasoner()
    out = r.step_tool_call([])
    assert out["kind"] == "text" and "tool_calls" not in out, out
    assert out["finish_reason"] == "length"
    assert "tool_calls" not in r.messages[-1], r.messages[-1]


def test_a_cut_turn_carries_no_reasoning_item_without_its_call(monkeypatch):
    """On the Responses wire a reasoning item rides only with the output item
    it preceded; a cut call is dropped, so its reasoning items go with it."""
    call = {"id": "c1", "type": "function"}
    call["function"] = {"name": "run", "arguments": '{"code": "go_to(prior_0'}
    item = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "E"}
    _serve(
        monkeypatch,
        reasoning="settled",
        finish_reason="length",
        tool_calls=[call],
        items=[item],
    )
    r = _reasoner()
    r.step_tool_call([])
    assert "reasoning_items" not in r.messages[-1], r.messages[-1]
