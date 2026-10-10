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

from maps.llm.endpoint_spec import EndpointSpec
from maps.llm.http import DegenerateCompletion, _RawResponse
from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import (
    MissionReply,
    MissionSpec,
    advance_reasoner_one_round,
)

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="inkling-small",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)


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
    return Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)


def test_collapse_in_reasoning_raises_and_leaves_history_clean(monkeypatch):
    """The measured shape: a few clean chars, then token 0 forever."""
    _serve(monkeypatch, reasoning="These" + "!" * 400, finish_reason="length")
    r = _reasoner()
    before = len(r.messages)
    with pytest.raises(DegenerateCompletion):
        r.step_tool_call([])
    # The whole point of raising in the agent rather than the loop: the run
    # must not reach the prefix, the next request, or the compactor.
    assert len(r.messages) == before


def test_collapse_in_content_channel_also_raises(monkeypatch):
    """Nothing about a NaN logit row is channel-specific.

    A collapse after ``</think>`` lands in ``content``, and the loop reads a
    non-empty prose reply as the mission's ONE model-owned ending -- so an
    uncaught one stops the mission on a reply of pure garbage, with no
    repair and no retry.
    """
    _serve(monkeypatch, content="!" * 400, finish_reason="length")
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


def test_no_stop_sequence_is_sent(monkeypatch):
    """`stop`/`no_stop_trim` are sglang-only and were removed.

    Trimming behaviour differs per engine -- sglang deletes matched stop
    text unless `no_stop_trim`, vLLM spells it `include_stop_str_in_output`,
    llama.cpp has neither -- so on any of them the stop would delete the run
    and DEFEAT this guard. `no_stop_trim` is also an unknown top-level field
    that a hosted OpenAI-compatible endpoint 400s on. Pinned because the
    cost of reintroducing it is silent and engine-dependent.
    """
    sent = _serve(monkeypatch, content="a reply")
    _reasoner().step_tool_call([])
    assert "stop" not in sent
    assert "no_stop_trim" not in sent


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


def test_loop_retries_a_collapse_instead_of_ending_the_mission(monkeypatch):
    """The mission survives a collapse: it is retriable, not terminal."""
    monkeypatch.setattr(
        "maps.search.parse_compile.reasoner_loop.time.sleep",
        lambda _s: None)

    turns = iter([
        DegenerateCompletion("collapse"),
        {"kind": "text", "content": "the goal is the flown object",
         "reasoning": ""},
    ] + [{"kind": "text", "content": "the goal is the flown object",
          "reasoning": ""}] * 4)

    def _step(tools):
        nxt = next(turns)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    r = _reasoner()
    r.step_tool_call = _step

    class _DummyCtx:
        pass

    reply = advance_reasoner_one_round(
        reasoner=r,
        user_message="find the car",
        ctx=_DummyCtx(),
        grounded_fids=set(),
        compute_namespace={"cands": [], "anchors": []},
        rid_tag="h0",
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000", "target": item["target_fid"],
                "pose_xy": [0.0, 0.0], "new_detections": 0},
            declare_concepts=lambda items: {"declared": []},
            flight_budget=4, turn_cap=16),
    )
    assert isinstance(reply, MissionReply)
    # The collapse was retried, not counted against the transient budget.
    assert reply.ended_by != "transient_failures"
