#!/usr/bin/env python
"""A collapsed completion is a SERVE fault, not the model's empty turn.

The two are identical on the wire -- no content, no tool call -- and the
loop used to read both as the model dropping a message: one repair nudge,
then the mission ends. On one sm_103 serve stack that cost a third of Cal
City episodes, because the serve was returning `!` (token id 0) x 40,000
on a fifth of xhigh requests while the same checkpoint on L40S returned
0/320 and the longest LEGITIMATE completion either serve produced was
~7.3k tokens against a 40,000 cap.

``Reasoner.step_tool_call`` raises ``DegenerateCompletion`` for that shape,
BEFORE the assistant turn is appended, so the garbage never enters the
prefix, and ``reasoner_loop._RETRIABLE_LLM_ERRORS`` names the class so the
existing transient path retries it.

The guard tests ONE thing: contentless, callless, and a repeated-token run
in either channel. It deliberately does NOT test ``finish_reason``. An
earlier version treated every ``finish_reason == "length"`` with no content
as degenerate, which is the CLIP -- a think span that ran long -- and that
arm was deterministic per prompt, so it did not cost one retry: six in a row
end the mission ``transient_failures``, which is dropped from SR as an infra
abort and re-queued for a full redo, forever. ``test_length_clip_*`` pins
that it no longer fires.

The vacuity floor is ``test_empty_turn_still_reads_as_a_text_turn``: the
model's own empty turn must still come back as a plain ``text`` turn and
must still append, or this guard would have swallowed the empty-turn
contract with every other test still green.

No live LLM. Run: ``pytest maps/search/tests/test_degenerate_completion.py``.
"""
from __future__ import annotations

import pytest

from maps.llm.http import DegenerateCompletion, _RawResponse
from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.tests.fakes import TEST_SPEC


def _serve(monkeypatch, *, content="", reasoning="", finish_reason="stop",
           tool_calls=()):
    """Pin the transport to one canned completion; return the sent kwargs."""
    sent: dict = {}

    def _fake(url, api_key, **kwargs):
        sent.update(kwargs)
        return _RawResponse({"choices": [{
            "finish_reason": finish_reason,
            "message": {"content": content, "reasoning_content": reasoning,
                        "tool_calls": list(tool_calls)},
        }]})

    monkeypatch.setattr(reasoner_agent, "_chat_completions_raw", _fake)
    return sent


def _reasoner() -> Reasoner:
    return Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC)


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
    """A think span that ran long is a CLIP, and clips keep their repair.

    This is the arm that was removed. Treating every contentless
    ``finish_reason="length"`` as degenerate is deterministic per prompt, so
    six in a row reach MAX_ROUND_TRANSIENT and end the mission
    ``transient_failures`` -- which analysis drops from SR as an infra abort
    and the driver re-queues for a full redo, forever. The empty-turn repair
    recovers 84% of these on the next turn; the guard must not steal them.
    """
    _serve(monkeypatch,
           reasoning="I need to weigh the two candidate rooftops. " * 40,
           finish_reason="length")
    r = _reasoner()
    before = len(r.messages)
    out = r.step_tool_call([])
    assert out["kind"] == "text"
    assert len(r.messages) == before + 1


def test_empty_turn_still_reads_as_a_text_turn(monkeypatch):
    """VACUITY FLOOR: the model's own empty turn is a DIFFERENT thing.

    Contentless, callless, ``finish_reason="stop"``, ordinary prose in the
    reasoning channel. 28/28 no-commit episodes were this shape and
    ``reasoner_loop``'s empty-turn branch owns it: one repair, then a
    model-owned ending. If the guard ever widens far enough to eat this,
    every other test here still passes and only this one fails.
    """
    _serve(monkeypatch, reasoning="I have nothing further to add.")
    r = _reasoner()
    before = len(r.messages)
    out = r.step_tool_call([])
    assert out["kind"] == "text"
    assert out["content"] == ""
    assert len(r.messages) == before + 1
