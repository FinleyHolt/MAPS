#!/usr/bin/env python
"""The mission's reasoning effort reaches the WIRE, on every request.

The field console's dial names an effort per mission
(``have_worlds/station/mode.py::EFFORT_CHOICES``) and the whole point is
that the mission's requests carry it in place of the arm's
``SEARCH_SAMPLING_REASONING_EFFORT``. The dial's other pins stop short of
the body: ``maps/llm/tests/test_sampling.py`` pins the kwargs builder,
and ``have_worlds/station/tests/test_mission_runner.py`` pins the session
handing the word to ``run_mission``. Nothing pinned the two passes
between them, so dropping either left the mission flying the arm's value
with every suite green and no crash row -- the failure this file exists
to make loud.

The vacuity floor is ``test_a_reasoner_with_no_word_sends_the_arms``: a
reasoner that names none must still send the environment's, or these
asserts would pass over a path that sends nothing at all.

No live LLM. Run: ``pytest maps/search/tests/test_effort_rides_every_request.py``.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from maps.llm.endpoint_spec import EndpointSpec
from maps.llm.http import _RawResponse
from maps.search.parse_compile import reasoner_agent
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import MissionReply, run_mission

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="qwen",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)


def _serve(monkeypatch) -> list[dict]:
    """Pin the transport to a plain text turn; return every sent body."""
    sent: list[dict] = []

    def _fake(url, api_key, **kwargs):
        sent.append(kwargs)
        return _RawResponse({"choices": [{
            "finish_reason": "stop",
            "message": {"content": "standing by", "reasoning_content": "",
                        "tool_calls": []},
        }]})

    monkeypatch.setattr(reasoner_agent, "_chat_completions_raw", _fake)
    return sent


def test_the_missions_word_rides_every_request_and_the_arms_does_not(monkeypatch):
    """Two rounds on ONE reasoner -- the field's later operator turns run
    through ``continue_mission`` on the same object -- and both carry the
    mission's word as a top-level string."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC,
                 reasoning_effort="0.3")
    r.append_user("find the truck")
    r.step_tool_call([])
    r.append_user("and the one north of it")
    r.step_tool_call([])
    assert [b["reasoning_effort"] for b in sent] == ["0.3", "0.3"]


def test_off_is_a_word_like_any_other(monkeypatch):
    """The dial's bottom rung is ``0.0``, which is a VALUE and not an
    absence: sent as the string, it turns thinking off. Read as falsy it
    would fall through to the arm's, which is the opposite request."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.99")
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC,
                 reasoning_effort="0.0")
    r.append_user("find the truck")
    r.step_tool_call([])
    assert sent[0]["reasoning_effort"] == "0.0"


def test_a_reasoner_with_no_word_sends_the_arms(monkeypatch):
    """VACUITY FLOOR: the environment is still the default, read at CALL
    time, so a rig select mid-day reaches a mission that named none."""
    monkeypatch.setenv("SEARCH_SAMPLING_REASONING_EFFORT", "0.9")
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    r.append_user("find the truck")
    r.step_tool_call([])
    assert sent[0]["reasoning_effort"] == "0.9"


def test_the_effort_reaches_no_served_byte(monkeypatch):
    """The CLAUDE.md law: the model is never told a budget fact. The dial
    is a sampling field on the envelope and appears nowhere in the bytes
    the model reads -- not the system prompt, not the tools, not a turn."""
    monkeypatch.delenv("SEARCH_SAMPLING_REASONING_EFFORT", raising=False)
    sent = _serve(monkeypatch)
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC,
                 reasoning_effort="0.3")
    r.append_user("find the truck")
    r.step_tool_call([{"type": "function", "function": {"name": "run"}}])
    served = json.dumps([sent[0]["messages"], sent[0]["tools"]]).lower()
    assert "effort" not in served and "0.3" not in served


def test_run_mission_hands_the_word_to_the_reasoner_it_builds(monkeypatch):
    """THE mission entry owns the Reasoner, so the station's word has to
    survive that construction; nothing downstream can put it back."""
    seen: dict = {}

    def _round(*, reasoner, **kw):
        seen["reasoner"] = reasoner
        return MissionReply(ended_by="reply", reply_text="standing by")

    monkeypatch.setattr(
        "maps.search.parse_compile.reasoner_loop.advance_reasoner_one_round",
        _round)
    _reply, reasoner = run_mission(
        instruction="find the truck",
        reasoner_url="http://unused",
        endpoint_spec=_TEST_SPEC,
        ctx=SimpleNamespace(),
        mission_spec=None,
        look_enabled=True,
        reasoning_effort="0.6",
    )
    assert seen["reasoner"] is reasoner
    assert reasoner.reasoning_effort == "0.6"
