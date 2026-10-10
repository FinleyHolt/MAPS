#!/usr/bin/env python
"""``Reasoner.compaction_events`` buffer + ``append_tool_result`` byte-format.

Two FROZEN contracts:

  * ``compaction_events`` is a per-instance ``field(default_factory=list)`` — the
    per-stop persistence point drains it to ``compaction_trace.jsonl``. If it were
    a class attribute, two reasoners in one turn's fanout would share (and
    cross-contaminate) one buffer; this pins that they do NOT.
  * ``append_tool_result`` serializes EXACTLY as the distillation renderer
    (``chat_render.tool_result_msg``): a ``str`` result passes verbatim; anything
    else is ``json.dumps(result, default=str)``; a value ``json.dumps`` cannot
    encode even with ``default=str`` (a non-str/non-scalar dict KEY) falls back to
    ``str(result)``. The student must train on the exact string the harness serves.

No network. Run:
``pytest maps/search/tests/test_reasoner_compaction_buffer.py``.
"""
from __future__ import annotations

import json

from maps.llm.endpoint_spec import EndpointSpec
from maps.search.parse_compile.reasoner_agent import Reasoner

# Offline stand-in for the server-discovered endpoint authority.
_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="qwen",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)


def _mk() -> Reasoner:
    # Cheap: __post_init__ only seeds the system prompt — no network.
    return Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)


# --------------------------------------------------------------------------- #
# compaction_events buffer                                                    #
# --------------------------------------------------------------------------- #


def test_new_reasoner_has_empty_buffer():
    r = _mk()
    assert r.compaction_events == []


def test_buffer_appends():
    r = _mk()
    r.compaction_events.append("evt-1")
    r.compaction_events.append("evt-2")
    assert r.compaction_events == ["evt-1", "evt-2"]


def test_two_reasoners_do_not_share_buffer():
    # default_factory=list, NOT a shared class attribute.
    r1, r2 = _mk(), _mk()
    assert r1.compaction_events is not r2.compaction_events
    r1.compaction_events.append("only-r1")
    assert r2.compaction_events == []


# --------------------------------------------------------------------------- #
# append_tool_result byte-format freeze                                        #
# --------------------------------------------------------------------------- #


def test_str_result_passes_verbatim():
    r = _mk()
    r.append_tool_result("c1", "analyze", "verbatim tool string")
    msg = r.messages[-1]
    assert msg == {"role": "tool", "tool_call_id": "c1",
                   "name": "analyze", "content": "verbatim tool string"}


def test_dict_result_is_json_dumps_default_str():
    r = _mk()
    result = {"len": 3, "fids": ["a", "b"], "score": 1.5}
    r.append_tool_result("c2", "analyze", result)
    # Exact byte-match including default separators (", " and ": ").
    assert r.messages[-1]["content"] == json.dumps(result, default=str)
    assert r.messages[-1]["content"] == '{"len": 3, "fids": ["a", "b"], "score": 1.5}'


def test_nonserializable_value_stringified_via_default_str():
    # A non-JSON value inside a dict is handled by default=str INSIDE json.dumps
    # (this is NOT the except-branch fallback — the whole result still encodes).
    class Weird:
        def __repr__(self):
            return "<Weird>"

    r = _mk()
    r.append_tool_result("c3", "analyze", {"obj": Weird()})
    assert r.messages[-1]["content"] == '{"obj": "<Weird>"}'


def test_unencodable_result_falls_back_to_str():
    # A tuple KEY makes json.dumps raise TypeError even with default=str
    # (default= only rescues VALUES, never keys) -> the except-branch str()
    # fallback fires.
    r = _mk()
    result = {(1, 2): "v"}
    r.append_tool_result("c4", "analyze", result)
    assert r.messages[-1]["content"] == str(result)
    assert r.messages[-1]["content"] == "{(1, 2): 'v'}"


def test_result_message_shape_is_tool_role():
    r = _mk()
    r.append_tool_result("cid-9", "select", {"ok": True})
    msg = r.messages[-1]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "cid-9"
    assert msg["name"] == "select"
    assert set(msg) == {"role", "tool_call_id", "name", "content"}
