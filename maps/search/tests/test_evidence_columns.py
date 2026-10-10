"""The episode's attached-column union fills, or it measures nothing.

``reasoner.evidence_columns`` accumulates every evidence-column name the
conversation's programs attached, across the whole episode. It is the offline read of what the agent
actually acquired against what the cue asked for -- the measurement that
showed acquisition running 58% on the deprescribed arm and ~69% when a
select-time attribution was required, while SR moved on neither.

The fold lives inside a ``try`` that swallows everything, so a union that
silently never fills would read exactly like a live one and every arm would
report 0% acquired. Assert a real column key reaches it, not just that the
set exists.
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

_CANDS = [
    {"fid": "d0", "xy": [0.0, 0.0], "class": "building", "rk": 3.0,
     "search_text": "building"},
    {"fid": "d1", "xy": [10.0, 0.0], "class": "building", "rk": 1.0,
     "search_text": "building"},
]
_GROUNDED = {"d0", "d1"}

# find -> order is a real ordering terminal, so its rows reach row_latest
# and the turn's attached-column census is non-empty.
_RANK_BLOCK = ('a = find("building")\n'
               'b = a.order("rk")\n'
               'go_to(b[0])')
_REPLY = "the middle building is b[0]."


class _DummyCtx:
    pass


def _turn(name, args):
    return {"kind": "tool_call", "content": "", "reasoning": "",
            "tool_calls": [{"id": "c0", "type": "function",
                            "function": {"name": name, "arguments": args}}]}


def _round(r, turns):
    # Trailing replies: a reply with no flight behind it is nudged rather
    # than accepted, so the script needs enough turns for the loop to
    # settle whichever way the block went.
    it = iter(list(turns) + [
        {"kind": "text", "content": _REPLY, "reasoning": ""}] * 4)
    r.step_tool_call = lambda tools: next(it)
    return advance_reasoner_one_round(
        reasoner=r,
        user_message="the middle building",
        ctx=_DummyCtx(),
        grounded_fids=set(_GROUNDED),
        compute_namespace={"cands": [dict(c) for c in _CANDS], "anchors": []},
        rid_tag="h0",
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000", "target": item["target_fid"],
                "pose_xy": [0.0, 0.0], "new_detections": 0},
            declare_concepts=lambda items: {"declared": []},
            flight_budget=4, turn_cap=8),
    )


def test_a_blocks_columns_reach_the_episode_union():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    _round(r, [_turn("run", {"code": _RANK_BLOCK}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    assert "rk" in r.evidence_columns, sorted(r.evidence_columns)


def test_the_union_is_episode_scoped_and_survives_the_next_turn():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    _round(r, [_turn("run", {"code": _RANK_BLOCK}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    first = set(r.evidence_columns)
    assert first, "the first block attached nothing"
    _round(r, [_turn("run", {"code": 'go_to("d0")'}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    assert first <= r.evidence_columns, (sorted(first),
                                         sorted(r.evidence_columns))


def test_a_mission_that_ran_no_program_leaves_an_empty_union_not_an_error():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=_TEST_SPEC)
    _round(r, [_turn("run", {"code": 'go_to("d0")'}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    assert r.evidence_columns == set(), r.evidence_columns
