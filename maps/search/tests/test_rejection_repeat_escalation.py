"""A re-served rejection carries its repeat count and the live census.

Corpus fact (episode birmingham_block_12__1_7181): the color-as-filter
rejection was re-served 195x consecutively -- 69 MB served over 205 turns --
because every serve was byte-identical, so nothing in the transcript told the
model it was looping. The repeat machinery exists to break that: the first
serve is the rejection verbatim, and from the second the served text carries
how many times the class has fired plus a census of what the live surface
actually holds (counts and computed column names, never a per-fid inventory,
which is token noise that scales with the surface).

``_loop_breaker_facts`` is reached through the REAL turn loop with a scripted
reasoner, because the counter and the census closure both live inside
``advance_reasoner_one_round`` and neither is importable on its own.

ponytail: the third-consecutive escalation to a compact repetition-state line
(``_repeated_rejection_state``, gated on ``_consec >= 3``) does not fire on
this surface -- the consecutive streak resets between serves, so a spiral gets
the full text plus a growing count forever rather than the compact form. The
count and census still break byte-identity, which is what the 7181 loop
lacked. Tested here is what the surface does; the gate is a separate fix.
"""
from __future__ import annotations

from maps.llm.endpoint_spec import EndpointSpec
from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import (
    MissionSpec, advance_reasoner_one_round)

_TEST_SPEC = EndpointSpec(
    base_url="http://unused",
    model_id="qwen36-35b-a3b",
    max_model_len=262144,
    tokenizer_name="qwen",
    output_reserve=40000,
)

# Colour never cuts, so this block is refused every time it is served.
_COLOR_FILTER_BLOCK = ('a = find("car")\n'
                       'b = a.filter(color="gray")\n'
                       'b')
_OK_BLOCK = 'a = find("car")\na'
# An op-SHAPE error: its own echoed line is the fix material, so it gets the
# short count line and never the census.
_SHAPE_BLOCK = 'a = find("car")\nb = a.order(\n'

_CANDS = [
    {"fid": "detected_0", "xy": [0.0, 0.0], "class": "car",
     "color": "gray", "color_conf": 0.8, "search_text": "car"},
    {"fid": "detected_1", "xy": [10.0, 0.0], "class": "car",
     "color": "blue", "color_conf": 0.7, "search_text": "car"},
]


class _DummyCtx:
    pass


def _run_turn(i: int, code: str) -> dict:
    return {"kind": "tool_call", "content": "", "reasoning": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "run",
                                         "arguments": {"code": code}}}]}


def drive_blocks(blocks: list[str]) -> list[dict]:
    """Run one REAL mission issuing ``blocks`` in order; return the served
    ``run`` tool-result payloads in order."""
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    turns = iter([_run_turn(i, b) for i, b in enumerate(blocks)]
                 # A reply with no flight behind it is nudged rather than
                 # accepted, so the tail has to outlast the abstain streak.
                 + [{"kind": "text", "content": "done.", "reasoning": ""}] * 5)
    r.step_tool_call = lambda tools: next(turns)
    served: list[dict] = []
    _orig = r.append_tool_result

    def _record(cid, name, payload):
        if name == "run":
            served.append(payload)
        return _orig(cid, name, payload)

    r.append_tool_result = _record
    advance_reasoner_one_round(
        reasoner=r,
        user_message="the gray car",
        ctx=_DummyCtx(),
        grounded_fids={"detected_0", "detected_1"},
        compute_namespace={"cands": [dict(c) for c in _CANDS],
                           "anchors": []},
        rid_tag="h0",
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000", "target": item["target_fid"],
                "pose_xy": [0.0, 0.0], "new_detections": 0},
            declare_concepts=lambda items: {"declared": []},
            flight_budget=4, turn_cap=16),
    )
    return served


def test_first_serve_is_the_rejection_verbatim():
    served = drive_blocks([_COLOR_FILTER_BLOCK])
    err = served[0]["error"]
    assert "colour never CUTS" in err
    assert "occurred" not in err  # no repetition machinery on first serve


def test_second_serve_carries_the_census_once():
    served = drive_blocks([_COLOR_FILTER_BLOCK] * 2)
    err2 = served[1]["error"]
    assert "colour never CUTS" in err2
    assert "occurred 2x." in err2
    assert "Live surface:" in err2  # the full census vehicle
    assert "Columns computed on rows so far" in err2


def test_the_count_keeps_climbing_so_no_two_serves_are_identical():
    # The 7181 loop's whole failure was byte-identity: 195 serves that said
    # exactly the same thing. The count is what breaks it.
    served = drive_blocks([_COLOR_FILTER_BLOCK] * 5)
    texts = [p["error"] for p in served]
    assert len(set(texts)) == len(texts), texts
    for k, payload in enumerate(served[1:], start=2):
        assert f"occurred {k}x." in payload["error"], payload["error"]


def test_no_fid_inventory_rides_a_repeat():
    # Counts only, so the served size does not scale with the surface.
    served = drive_blocks([_COLOR_FILTER_BLOCK] * 4)
    for payload in served[1:]:
        err = payload["error"]
        assert "detected_0" not in err, err
        assert len(err) < 700, len(err)


def test_a_successful_block_does_not_reset_the_class_count():
    served = drive_blocks(
        [_COLOR_FILTER_BLOCK] * 2 + [_OK_BLOCK] + [_COLOR_FILTER_BLOCK])
    assert "error" not in served[2], served[2]
    err4 = served[3]["error"]
    assert "colour never CUTS" in err4
    assert "occurred 3x." in err4


def test_shape_errors_keep_the_short_count_line():
    served = drive_blocks([_SHAPE_BLOCK] * 3)
    assert "syntax error" in served[0]["error"].lower(), served[0]
    for k, payload in enumerate(served[1:], start=2):
        err = payload["error"]
        assert f"occurred {k}x." in err, err
        # a shape error's own echoed line is the fix material; the census is
        # pure token noise beside it
        assert "Live surface:" not in err, err
