from __future__ import annotations

from maps.search.parse_compile.reasoner_agent import Reasoner
from maps.search.parse_compile.reasoner_loop import (
    MissionSpec,
    advance_reasoner_one_round,
)
from maps.search.tests.fakes import TEST_SPEC, bare_ctx

_CANDS = [
    {"fid": "d0", "xy": [0.0, 0.0], "class": "building", "rk": 3.0,
     "search_text": "building"},
    {"fid": "d1", "xy": [10.0, 0.0], "class": "building", "rk": 1.0,
     "search_text": "building"},
]

# find -> order is a real ordering terminal, so its rows reach row_latest
# and the turn's attached-column census is non-empty.
_RANK_BLOCK = ('a = find("building")\n'
               'b = a.order("rk")\n'
               'go_to(b[0])')
_REPLY = "the middle building is b[0]."


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
        ctx=bare_ctx(),
        compute_namespace={"cands": [dict(c) for c in _CANDS], "anchors": []},
        rid_tag="h0",
        mission_spec=MissionSpec(
            fly=lambda item: {
                "stop_id": "stop_000", "target": item["target_fid"],
                "pose_xy": [0.0, 0.0], "new_detections": 0},
            declare_concepts=lambda items: {"declared": []},
            flight_budget=4, turn_cap=8),
    )


def test_the_union_is_episode_scoped_and_survives_the_next_turn():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC)
    _round(r, [_turn("run", {"code": _RANK_BLOCK}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    first = set(r.evidence_columns)
    assert "rk" in first, sorted(first)
    _round(r, [_turn("run", {"code": 'go_to("d0")'}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    assert first <= r.evidence_columns, (sorted(first),
                                         sorted(r.evidence_columns))


def test_a_mission_that_ran_no_program_leaves_an_empty_union_not_an_error():
    r = Reasoner(reasoner_url="http://unused", endpoint_spec=TEST_SPEC)
    _round(r, [_turn("run", {"code": 'go_to("d0")'}),
               {"kind": "text", "content": _REPLY, "reasoning": ""}])
    assert r.evidence_columns == set(), r.evidence_columns
