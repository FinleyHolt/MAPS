#!/usr/bin/env python
"""Folding spent turns into the agent's own code, append-only across fires."""
from __future__ import annotations

import json

import pytest

from maps.search.analyze_scene import compactor as C
from maps.search.analyze_scene.compactor import (
    _SCRIPT_BLOCK_HEADER,
    _drop_spent_turns,
    _is_script_block,
    _run_code_of,
    _turn_units,
    _validate_invariants,
)


def _content_str(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _charcount(messages, *, model, tools=None):
    return sum(len(_content_str(m)) // 4 for m in messages)


def _turn(tcid: str, code: str, result_size: int = 4000) -> list[dict]:
    """One [assistant run-call, tool result] unit, the mission surface's shape."""
    return [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": tcid, "type": "function",
                         "function": {"name": "run",
                                      "arguments": json.dumps({"code": code})}}]},
        {"role": "tool", "tool_call_id": tcid, "name": "run",
         "content": json.dumps({"result": {"rows": "x" * result_size}})},
    ]


def _transcript(n: int) -> list[dict]:
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "find the church"},
    ]
    for i in range(n):
        msgs += _turn(f"t{i}", f"r{i} = find('thing {i}')  # step {i}")
    return msgs


@pytest.fixture(autouse=True)
def _fake_estimator(monkeypatch):
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)


def test_the_block_carries_the_agent_s_own_code_verbatim() -> None:
    msgs = _transcript(6)
    out, folded, freed = _drop_spent_turns(
        msgs, shed_tokens=2000, model="m")
    assert folded >= 1 and freed > 0
    block = next(m for m in out if _is_script_block(m))
    assert _SCRIPT_BLOCK_HEADER in block["content"]
    # Verbatim, comments included -- the comment IS the agent's rationale, and
    # a paraphrase would lose exactly that.
    assert "r0 = find('thing 0')  # step 0" in block["content"]


def test_the_protected_head_and_the_live_tail_are_never_folded() -> None:
    msgs = _transcript(6)
    out, folded, _ = _drop_spent_turns(msgs, shed_tokens=10**6, model="m")
    assert out[0]["role"] == "system" and out[0]["content"] == "SYS"
    assert out[1]["content"] == "find the church", (
        "the operator instruction is the ONE user message a bench episode has"
    )
    # The last two units stay verbatim: the model is mid-derivation there.
    assert sum(1 for m in out if m.get("role") == "tool") >= 2


def test_a_turn_and_its_results_move_as_one_unit() -> None:
    msgs = _transcript(6)
    out, folded, _ = _drop_spent_turns(msgs, shed_tokens=10**6, model="m")
    assert folded >= 1
    _validate_invariants(out)


def test_a_block_is_never_itself_folded() -> None:
    msgs = _transcript(8)
    once, _, _ = _drop_spent_turns(msgs, shed_tokens=2000, model="m")
    twice, _, _ = _drop_spent_turns(once, shed_tokens=10**6, model="m")
    assert any(_is_script_block(m) for m in twice)
    assert "r0 = find('thing 0')" in next(
        m for m in twice if _is_script_block(m))["content"], (
        "folding a block would lose the code it was built to preserve"
    )


def test_nothing_to_fold_is_the_identity() -> None:
    msgs = _transcript(2)  # both units are the protected tail
    out, folded, freed = _drop_spent_turns(msgs, shed_tokens=10**6, model="m")
    assert out is msgs and folded == 0 and freed == 0


def test_run_code_survives_a_malformed_argument() -> None:
    """An unreadable action still says one happened."""
    msg = {"role": "assistant", "tool_calls": [
        {"id": "x", "function": {"name": "run", "arguments": "not json"}}]}
    assert _run_code_of(msg) == ["not json"]


def test_turn_units_group_multi_result_calls() -> None:
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a", "function": {"name": "run", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "run", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "user", "content": "next"},
    ]
    assert _turn_units(msgs) == [(0, 3)]
