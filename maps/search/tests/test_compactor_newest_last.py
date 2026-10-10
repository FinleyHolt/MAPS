#!/usr/bin/env python
"""Tier-3 degrade order: the NEWEST analyze dump trims dead-last.

The newest dump is the round's live evidence — starving it forces the model
to decide from memory (the choosable-miss forensics found verification
queries returning as fids+census stubs at exactly the deciding moment).
``_degrade_oversized_tool_dumps`` therefore trims stale dumps largest-first
and touches the newest only when every stale dump together cannot pay the
budget (feasibility unchanged).

Pure functions; the token estimator is stubbed to a character count so the
order is asserted deterministically. Run:
``pytest maps/search/tests/test_compactor_newest_last.py``.
"""
from __future__ import annotations

import json

import maps.search.analyze_scene.compactor as compactor
from maps.search.analyze_scene.compactor import _degrade_oversized_tool_dumps


def _dump(fids: list[str], pad: int) -> str:
    rows = "\n".join(f"| {i} | {f} | car | gray | 0.5 |" + "x" * pad
                     for i, f in enumerate(fids))
    return json.dumps({"results": {"r0": {
        "len": len(fids), "fids": fids, "by_class": {"car": len(fids)},
        "units": "d_/dc_/along_/off_/vx_/vy_ in meters",
        "rows": rows,
    }}})


def _messages() -> list[dict]:
    """system/user + three analyze dumps: big-old, mid-stale, small-NEWEST."""
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "name": "analyze", "tool_call_id": "a",
         "content": _dump(["d0", "d1"], pad=400)},   # oldest, biggest
        {"role": "tool", "name": "analyze", "tool_call_id": "b",
         "content": _dump(["d2", "d3"], pad=200)},   # stale, mid
        {"role": "tool", "name": "analyze", "tool_call_id": "c",
         "content": _dump(["d4", "d5"], pad=50)},    # NEWEST, smallest
    ]


def _char_estimator(messages, *, model, tools=None):
    return sum(len(str(m.get("content") or "")) for m in messages)


def test_newest_survives_when_stale_trims_suffice(monkeypatch):
    monkeypatch.setattr(compactor, "_estimate_input_tokens", _char_estimator)
    msgs = _messages()
    # Budget reachable by trimming the two stale dumps alone.
    target = _char_estimator(msgs, model="m") - 900
    out, n = _degrade_oversized_tool_dumps(
        msgs, model="m", tools=None, target_input_tokens=target)
    assert n >= 1
    newest = json.loads(out[4]["content"])
    assert "rows" in newest["results"]["r0"], "newest dump must keep its rows"
    oldest = json.loads(out[2]["content"])
    assert "rows" not in oldest["results"]["r0"], "biggest stale dump trims first"


def test_newest_still_degrades_when_budget_demands(monkeypatch):
    monkeypatch.setattr(compactor, "_estimate_input_tokens", _char_estimator)
    msgs = _messages()
    out, n = _degrade_oversized_tool_dumps(
        msgs, model="m", tools=None, target_input_tokens=10)
    assert n == 3, "feasibility unchanged: every dump degrades when needed"
    for i in (2, 3, 4):
        head = json.loads(out[i]["content"])["results"]["r0"]
        assert "rows" not in head
        assert head["fids"], "no candidate ever leaves the pool"
