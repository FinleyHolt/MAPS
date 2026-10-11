#!/usr/bin/env python
"""Tool-protocol closure, checked after the compactor's content rewrites."""
from __future__ import annotations

import pytest

from maps.search.analyze_scene.compactor import (
    CompactionInvariantError,
    _validate_invariants,
)


def test_validate_raises_on_orphan_tool_result():
    with pytest.raises(CompactionInvariantError) as ei:
        _validate_invariants([
            {"role": "tool", "tool_call_id": "nope", "name": "n",
             "content": "x"},
        ])
    assert "orphan tool" in str(ei.value)
    assert "nope" in str(ei.value)


def test_validate_raises_when_the_call_comes_after_its_result():
    """Closure is DIRECTIONAL: the producing call has to be earlier, so a reordering
    that floats a tool result above its call is caught."""
    with pytest.raises(CompactionInvariantError):
        _validate_invariants([
            {"role": "tool", "tool_call_id": "a1", "name": "analyze",
             "content": "ok"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a1"}]},
        ])


def test_a_call_with_no_result_is_caught() -> None:
    """Coverage: the half that only became reachable when a step started dropping
    messages."""
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a1"}]},
        {"role": "user", "content": "the result for a1 is gone"},
    ]
    with pytest.raises(CompactionInvariantError) as ei:
        _validate_invariants(msgs)
    assert "unanswered" in str(ei.value)
    assert "a1" in str(ei.value)
