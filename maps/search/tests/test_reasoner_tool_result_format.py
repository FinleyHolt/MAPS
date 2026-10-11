#!/usr/bin/env python
"""``Reasoner.append_tool_result`` byte-format."""
from __future__ import annotations

import json

from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.tests.fakes import TEST_SPEC


# Offline stand-in for the server-discovered endpoint authority.
def _mk() -> Reasoner:
    # Cheap: __post_init__ only seeds the system prompt — no network.
    return Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC)


# --------------------------------------------------------------------------- #
# append_tool_result byte-format freeze                                        #
# --------------------------------------------------------------------------- #


def test_dict_result_is_json_dumps_default_str():
    r = _mk()
    result = {"len": 3, "fids": ["a", "b"], "score": 1.5}
    r.append_tool_result("c2", "analyze", result)
    # Exact byte-match including default separators (", " and ": ").
    assert r.messages[-1]["content"] == json.dumps(result, default=str)
    assert r.messages[-1]["content"] == '{"len": 3, "fids": ["a", "b"], "score": 1.5}'


def test_nonserializable_value_stringified_via_default_str():
    # A non-JSON value inside a dict is handled by default=str INSIDE json.dumps.
    class Weird:
        def __repr__(self):
            return "<Weird>"

    r = _mk()
    r.append_tool_result("c3", "analyze", {"obj": Weird()})
    assert r.messages[-1]["content"] == '{"obj": "<Weird>"}'
