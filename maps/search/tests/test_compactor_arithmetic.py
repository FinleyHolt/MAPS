#!/usr/bin/env python
"""Arithmetic + tier control-flow of ``compactor.compact_transcript`` and the
``EndpointSpec`` budget derivation that feeds it.

Offline + hermetic: the real ``_estimate_input_tokens`` loads the Qwen HF
tokenizer (network). It is monkeypatched with a deterministic char-count fake —
the CONTROL FLOW and the token arithmetic are what these pin, not HF. The
clear-to-census compactor makes NO LLM call on this path, so nothing is patched
but the estimator. The estimator is length-sensitive: it SHRINKS when a verbose
``rows`` render (or a per-fid ``colors`` table) is dropped, so the tier chain —
(1) clear stale dumps to fids+census+colors, (2) strip color tables, (3)
degrade the live dump, (4) CompactionInfeasible — is exercised faithfully.

The budget is CALLER-SUPPLIED (``target_input_tokens``), derived from the
server-discovered ``EndpointSpec.input_budget`` in production — there is no
window env knob to steer.

Run: ``pytest maps/search/tests/test_compactor_arithmetic.py`` (the
suite runs ``-n auto``; every test is independent + env-scoped).
"""
from __future__ import annotations

import json

import pytest

from maps.llm.endpoint_spec import EndpointSpec
from maps.search.analyze_scene import compactor as C
from maps.search.analyze_scene.compactor import (
    CompactionError,
    CompactionEvent,
    CompactionInfeasible,
    compact_transcript,
)


# --------------------------------------------------------------------------- #
# Deterministic token-estimate fake                                           #
# --------------------------------------------------------------------------- #


def _content_str(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c, default=str)


def _charcount(messages, *, model, tools=None):
    """A stand-in for ``_estimate_input_tokens``: ~4 chars/token over the
    serialized message contents. Deterministic, offline, and — crucially — it
    SHRINKS when a verbose ``rows`` render is stubbed OR a per-fid ``colors``
    table is stripped, so the tier arithmetic is exercised faithfully."""
    return sum(len(_content_str(m)) // 4 for m in messages)


def _const(value):
    def _fake(messages, *, model, tools=None):
        return value
    return _fake


# --------------------------------------------------------------------------- #
# Transcript builders — nested LIVE analyze payload shape                     #
# --------------------------------------------------------------------------- #


def _color_grid(color_rows, *, filler: int = 0) -> str:
    """A realistic model-facing markdown grid (``| idx | fid | class | color |
    color_conf | xy |``) — the same header/columns ``geom_program`` renders, so
    ``_rows_color_entries`` parses ``fid=color[@conf]`` out of it. ``filler``
    appends a bulky NON-pipe line (dropped wholesale with ``rows``; ignored by
    the color parser) to inflate the dump without touching the color table."""
    header = "| idx | fid | class | color | color_conf | xy |"
    sep = "| --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for i, (fid, col, conf) in enumerate(color_rows):
        lines.append(f"| {i} | {fid} | car | {col} | {conf} | 0.0, 0.0 |")
    grid = "\n".join(lines)
    if filler:
        grid += "\n" + ("z" * filler)
    return grid


def _analyze_dump(tcid: str, fids, color_rows, *, filler: int = 0) -> dict:
    """One ``analyze`` tool-result message in the LIVE nested payload shape
    ``{"results": {"r0": head}}`` — the shape ``_iter_op_heads`` resolves
    through its ``results`` wrapper."""
    head = {
        "len": len(fids),
        "fids": list(fids),
        "units": "d_/dc_/along_/off_/vx_/vy_ in meters",
        "rows": _color_grid(color_rows, filler=filler),
    }
    payload = {"results": {"r0": head}}
    return {"role": "tool", "tool_call_id": tcid, "name": "analyze",
            "content": json.dumps(payload)}


def _assistant_call(tcid: str) -> dict:
    """The assistant tool_call turn that PRODUCES ``tcid`` — every analyze
    tool-result needs one earlier so ``_validate_invariants`` closure holds."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": tcid, "type": "function",
                            "function": {"name": "analyze", "arguments": "{}"}}]}


def _transcript(dumps: list[dict]) -> list[dict]:
    """[system, (assistant_call, analyze_dump)*, user] — tool-protocol closed
    and > the 4-msg cheap gate."""
    msgs: list[dict] = [{"role": "system", "content": "SYS"}]
    for d in dumps:
        msgs.append(_assistant_call(d["tool_call_id"]))
        msgs.append(d)
    msgs.append({"role": "user", "content": "u1"})
    return msgs


# A 3-row color grid: a confident white, a below-floor '?'-masked red, a blue.
_CR = [("d0", "white", "0.8"), ("d1", "red?", "0.0"), ("d2", "blue", "0.5")]


def _analyze_indices(msgs):
    return [i for i, m in enumerate(msgs)
            if m.get("role") == "tool" and m.get("name") == "analyze"]


def _roles(msgs):
    return [m.get("role") for m in msgs]


def _tool_call_ids(msgs):
    return [m.get("tool_call_id") for m in msgs]


# --------------------------------------------------------------------------- #
# The budget feeding target_input_tokens: EndpointSpec window arithmetic       #
# --------------------------------------------------------------------------- #


def _spec(max_model_len: int) -> EndpointSpec:
    return EndpointSpec(
        base_url="http://unused/v1",
        model_id="qwen36-35b-a3b",
        max_model_len=max_model_len,
        tokenizer_name="Qwen/Qwen3.6-35B-A3B",
        output_reserve=min(40000, max_model_len // 4),
    )


def test_deployed_window_budget_is_window_minus_reserve_minus_margin():
    s = _spec(262144)
    assert s.output_reserve == 40000
    assert s.input_budget == 262144 - 40000 - 1024
    assert s.max_output_tokens == 40000


def test_small_window_derivation_no_negative_budget():
    # The Gemma-swap case that produced the 41% 400 storm: 32768 window.
    s = _spec(32768)
    assert s.output_reserve == 8192           # min(40000, 32768 // 4)
    assert s.input_budget == 32768 - 8192 - 1024
    assert s.input_budget > 8000              # discover()'s case-f holds
    assert s.max_output_tokens == 8192        # floor == reserve at 32k


# --------------------------------------------------------------------------- #
# Unmeasurable prompt: offline passes through, LIVE raises                     #
# --------------------------------------------------------------------------- #


def _five_messages():
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]


class _NoToolsTemplate:
    """A tokenizer whose chat template refuses the ``tools`` kwarg."""

    def apply_chat_template(self, messages, tools=None, **_kw):
        if tools is not None:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'tools'")
        return "rendered"

    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


def test_sizing_a_prompt_raises_on_a_render_failure(monkeypatch):
    monkeypatch.setattr(C, "_tokenizer", lambda m: _NoToolsTemplate())
    with pytest.raises(TypeError, match="tools"):
        C._estimate_input_tokens(_five_messages(), model="m", tools=[{"type": "function"}])


# --------------------------------------------------------------------------- #
# Cheap gate: <= 4 messages -> identity passthrough (same id), no estimate     #
# --------------------------------------------------------------------------- #


def test_cheap_gate_identity_below_msg_count(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    # Estimate would report WAY over budget, but the <=4-msg cheap gate
    # short-circuits before it is ever consulted.
    monkeypatch.setattr(C, "_estimate_input_tokens", _const(10**9))
    events: list[CompactionEvent] = []
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    out = compact_transcript(msgs, target_input_tokens=10, model="qwen",
                             log_fn=events.append)
    assert out is msgs       # same object: byte-identical APC preserved
    assert events == []      # cheap gate never logs an event


# --------------------------------------------------------------------------- #
# Below budget: identity (same id) + a fired=False headroom event             #
# --------------------------------------------------------------------------- #


def test_below_budget_identity_and_headroom_event(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    monkeypatch.setattr(C, "_estimate_input_tokens", _const(50))
    events: list[CompactionEvent] = []
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    out = compact_transcript(msgs, target_input_tokens=1000, model="qwen",
                             site="reasoner.step", log_fn=events.append)
    assert out is msgs
    assert len(events) == 1
    ev = events[0]
    assert ev.fired is False and ev.infeasible is False
    assert ev.input_tokens == 50
    assert ev.target_input_tokens == 1000
    assert ev.site == "reasoner.step"


# --------------------------------------------------------------------------- #
# TIER 1: clear every STALE analyze dump to fids+census+colors; live verbatim  #
# --------------------------------------------------------------------------- #


def test_tier1_clears_stale_dumps_keeps_color_live_verbatim(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    events: list[CompactionEvent] = []
    # 3 nested-shape analyze dumps, each with a big verbose rows render. Clearing
    # the 2 STALE ones (rows -> fids+census+colors) fits under 8000; the live
    # (last) dump stays verbatim.
    dumps = [_analyze_dump(f"t{i}", ["d0", "d1", "d2"], _CR, filler=20000)
             for i in range(3)]
    msgs = _transcript(dumps)
    input_est = _charcount(msgs, model="qwen")
    assert input_est > 8000  # sanity: over budget
    out = compact_transcript(msgs, target_input_tokens=8000, model="qwen",
                             site="reasoner.step_tool_call", log_fn=events.append)

    # A NEW list is returned when fired; message count/order/roles/links frozen.
    assert out is not msgs
    assert len(out) == len(msgs)
    assert _roles(out) == _roles(msgs)
    assert _tool_call_ids(out) == _tool_call_ids(msgs)

    ai = _analyze_indices(out)
    stale_idxs, live_idx = ai[:-1], ai[-1]
    assert len(stale_idxs) == 2

    for i in stale_idxs:
        head = json.loads(out[i]["content"])["results"]["r0"]
        # render keys gone ...
        assert "rows" not in head and "units" not in head and "grid" not in head
        # ... fids + census + hint + colors present.
        assert head["fids"] == ["d0", "d1", "d2"]      # no candidate leaves the pool
        assert head["len"] == 3
        assert "fids preserved" in head["hint"]
        assert "colors" in head
        assert "d0=white@0.8" in head["colors"]        # color signal survives
        assert "d1=red?@0.0" in head["colors"]         # the '?' uncertainty mask intact

    # LIVE (last) analyze dump untouched — byte-identical to input.
    assert out[live_idx]["content"] == msgs[live_idx]["content"]

    assert len(events) == 1
    ev = events[0]
    assert ev.fired is True and ev.infeasible is False
    assert ev.cleared_dumps == 2
    assert ev.color_tables_kept == 2       # tier 2 not reached: both kept
    assert ev.degraded_dumps == 0           # tier 3 not reached
    assert ev.input_tokens == input_est
    assert ev.target_input_tokens == 8000
    assert ev.output_tokens <= 8000


# --------------------------------------------------------------------------- #
# TIER 2: still over after clearing -> strip color tables oldest-first        #
# --------------------------------------------------------------------------- #


def test_tier2_strips_color_tables_when_still_over(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    events: list[CompactionEvent] = []
    # Two STALE dumps whose bulk IS the color table (50 fids each, no filler),
    # plus a small live dump. Clearing to fids+census+colors (tier 1) still
    # overflows target=700, so the oldest color table must be stripped (tier 2);
    # one strip brings it under, so exactly one table survives.
    big_cr = [(f"d{i}", "reddishbrown?", "0.4") for i in range(50)]
    stale = [_analyze_dump(f"t{i}", [f"d{i}" for i in range(50)], big_cr)
             for i in range(2)]
    live = _analyze_dump("tL", ["z0"], [("z0", "green", "0.9")])
    msgs = _transcript(stale + [live])
    out = compact_transcript(msgs, target_input_tokens=700, model="qwen",
                             log_fn=events.append)

    assert out is not msgs
    assert _roles(out) == _roles(msgs)
    assert _tool_call_ids(out) == _tool_call_ids(msgs)

    ev = events[0]
    assert ev.fired is True and ev.infeasible is False
    assert ev.cleared_dumps == 2
    # A color table was stripped: kept count strictly decreased, no tier-3 hit.
    assert ev.color_tables_kept < ev.cleared_dumps
    assert ev.color_tables_kept == 1
    assert ev.degraded_dumps == 0
    assert ev.output_tokens <= 700

    # The surviving-with-colors stub count matches color_tables_kept exactly;
    # the OLDEST stale stub is the one that lost its colors.
    stale_idxs = _analyze_indices(out)[:-1]
    with_colors = [
        i for i in stale_idxs
        if "colors" in json.loads(out[i]["content"])["results"]["r0"]
    ]
    assert len(with_colors) == ev.color_tables_kept
    oldest = stale_idxs[0]
    assert "colors" not in json.loads(out[oldest]["content"])["results"]["r0"]
    # Every stale stub still lost its rows regardless of color fate.
    for i in stale_idxs:
        head = json.loads(out[i]["content"])["results"]["r0"]
        assert "rows" not in head
        assert head["fids"]  # pool intact


# --------------------------------------------------------------------------- #
# TIER 3: still over after color strip -> degrade the LIVE dump largest-first  #
# --------------------------------------------------------------------------- #


def test_tier3_degrades_live_dump_when_still_over(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    events: list[CompactionEvent] = []
    # Two small stale dumps + one LIVE dump with a monster rows render. Clearing
    # + color-stripping the stale dumps can't help (the live dump dominates), so
    # the live dump itself must be degraded (tier 3).
    small_cr = [("a0", "white", "0.8"), ("a1", "red?", "0.0")]
    stale = [_analyze_dump(f"t{i}", ["a0", "a1"], small_cr) for i in range(2)]
    live = _analyze_dump("tL", ["z0"], [("z0", "green", "0.9")], filler=32000)
    msgs = _transcript(stale + [live])
    out = compact_transcript(msgs, target_input_tokens=500, model="qwen",
                             log_fn=events.append)

    assert out is not msgs
    assert _roles(out) == _roles(msgs)
    assert _tool_call_ids(out) == _tool_call_ids(msgs)

    ev = events[0]
    assert ev.fired is True and ev.infeasible is False
    assert ev.cleared_dumps == 2
    assert ev.degraded_dumps >= 1           # the live dump was degraded
    assert ev.output_tokens <= 500

    # The LIVE dump lost its rows (degraded to plain fids+census) but kept fids.
    live_idx = _analyze_indices(out)[-1]
    live_head = json.loads(out[live_idx]["content"])["results"]["r0"]
    assert "rows" not in live_head and "units" not in live_head
    assert live_head["fids"] == ["z0"]


# --------------------------------------------------------------------------- #
# TIER 4: nothing clearable -> CompactionInfeasible + an infeasible event      #
# --------------------------------------------------------------------------- #


def test_tier4_infeasible_when_nothing_clearable(monkeypatch):
    monkeypatch.setenv("SEARCH_COMPACT_ENABLE", "1")
    monkeypatch.setattr(C, "_estimate_input_tokens", _charcount)
    events: list[CompactionEvent] = []
    # Over budget but NO analyze dumps (and no trimmable tool dumps at all): the
    # residue is head/prose mass the design deliberately never touches, so tiers
    # 1-3 clear/degrade nothing and the call aborts infeasible.
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U" * 4000},
        {"role": "assistant", "content": "A" * 4000},
        {"role": "user", "content": "V" * 4000},
        {"role": "assistant", "content": "B" * 4000},
        {"role": "user", "content": "done"},
    ]
    input_est = _charcount(msgs, model="qwen")
    with pytest.raises(CompactionInfeasible) as ei:
        compact_transcript(msgs, target_input_tokens=100, model="qwen",
                           site="reasoner.step", log_fn=events.append)
    assert "still exceeds" in str(ei.value)
    assert "cleared 0" in str(ei.value) and "degraded 0" in str(ei.value)

    assert len(events) == 1
    ev = events[0]
    assert ev.infeasible is True and ev.fired is False
    assert ev.reason and "still exceeds" in ev.reason
    assert ev.input_tokens == input_est
    assert ev.target_input_tokens == 100
