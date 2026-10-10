#!/usr/bin/env python
"""Tool-protocol closure, checked after the compactor's content rewrites.

``_validate_invariants`` rejects a transcript whose tool protocol is broken, so
a compacted list never POSTs a tool result with no producing call -- the server
400s on it.

The sweep checks CLOSURE (a result has an earlier call) AND COVERAGE (a call
has a result). Coverage became reachable when step 4 started folding spent turns
into a script block: every step before it rewrites content bytes only, leaving
count, order and roles invariant, so neither violation could arise. A
half-dropped turn can arise now, and coverage is the half the server would
otherwise catch as a 400 mid-episode.

Pure functions, no network. Run:
``pytest maps/search/tests/test_compactor_invariants.py``.
"""
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
    """Closure is DIRECTIONAL: the producing call has to be earlier, so a
    reordering that floats a tool result above its call is caught."""
    with pytest.raises(CompactionInvariantError):
        _validate_invariants([
            {"role": "tool", "tool_call_id": "a1", "name": "analyze",
             "content": "ok"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "a1"}]},
        ])


def test_validate_passes_on_a_closed_tool_exchange():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "instruction"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a1"}]},
        {"role": "tool", "tool_call_id": "a1", "name": "analyze",
         "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    _validate_invariants(msgs)


def test_a_call_with_no_result_is_caught() -> None:
    """Coverage: the half that only became reachable when a step started
    dropping messages.

    This test asserted the OPPOSITE until step 4 landed, and said so -- the gap
    was pinned as a decision, with a note that a span-dropping step would be the
    thing that inverts it. Folding spent turns makes a half-dropped turn
    constructible, and the alternative to catching it here is catching it as a
    400 mid-episode with the request already sent.
    """
    msgs = [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "a1"}]},
        {"role": "user", "content": "the result for a1 is gone"},
    ]
    with pytest.raises(CompactionInvariantError) as ei:
        _validate_invariants(msgs)
    assert "unanswered" in str(ei.value)
    assert "a1" in str(ei.value)
