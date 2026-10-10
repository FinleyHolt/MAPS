"""The mission contract: a block flies on the call; the plain reply ends it.

Drives ``advance_reasoner_one_round(mission_spec=...)`` with a scripted
reasoner — no serve, no SAM. This file owns the harness the surface's other
test files drive through (``FakeReasoner``, ``_ctx``, ``_CANDS``, ``_drive``,
``_tool_results``) and pins the loop-level semantics: the ONE model-owned
ending is a clean no-tool-call reply, a reply with nothing flown behind it is
repaired rather than accepted, an unknown tool name is answered and never
executed, and the liveness bounds behave with no budget fact ever served.

The library's own semantics — what each call does to the world and what comes
back — are pinned in ``test_code_surface_arm`` and ``test_python_surface``.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from maps.search.analyze_scene.geom_program import _BROAD_HEAD_ROWS as H
from maps.search.parse_compile.reasoner_loop import (
    MissionReply,
    MissionSpec,
    advance_reasoner_one_round,
    _mission_sanitize_moves,
)


class FakeReasoner:
    """Scripted turn source. A turn is either a plain string (a text reply)
    or a list of (tool_name, args_dict) calls emitted together."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.exhausted = 0
        self.messages = [{"role": "system", "content": "sys"}]
        self.compaction_infeasible = False
        self.evidence_columns = set()
        self.combined_tag = None
        self.model = "test"

    def append_user(self, content):
        self.messages.append({"role": "user", "content": content})

    def append_repair(self, message):
        self.messages.append(
            {"role": "user", "content": f"[Repair] {message}"})

    def append_tool_result(self, cid, name, result):
        content = result if isinstance(result, str) else json.dumps(result)
        self.messages.append({"role": "tool", "tool_call_id": cid,
                              "name": name, "content": content})

    def append_vision_user(self, text, jpegs):
        self.messages.append({"role": "user", "content": text})

    def step_tool_call(self, tools):
        # A REPAIR COSTS A TURN, and a fixture that scripts only the turns it
        # cares about should not have to script an answer to the loop's own
        # nudge. On exhaustion, reply once more -- a plain reply is the
        # mission's ending, so the loop RETURNS rather than spinning.
        # ``exhausted`` is kept so a test that means to pin an exact turn
        # count still can, and a genuine runaway surfaces as the assertion
        # below rather than as a confusing crash mid-script.
        if not self.turns:
            self.exhausted += 1
            assert self.exhausted <= 2, "loop did not return after two replies"
            self.messages.append({"role": "assistant", "content": ""})
            return {"kind": "text", "content": "", "reasoning": "",
                    "finish_reason": "stop"}
        t = self.turns.pop(0)
        if isinstance(t, str):
            self.messages.append({"role": "assistant", "content": t})
            return {"kind": "text", "content": t, "reasoning": "",
                    "finish_reason": "stop"}
        calls = [{"id": f"c{len(self.messages)}_{i}", "type": "function",
                  "function": {"name": n, "arguments": json.dumps(a)}}
                 for i, (n, a) in enumerate(t)]
        self.messages.append({"role": "assistant", "content": "",
                              "tool_calls": calls})
        return {"kind": "tool_call", "tool_calls": calls, "content": "",
                "reasoning": "", "finish_reason": "tool_calls"}




def _ctx():
    return SimpleNamespace(graph=None, cell_size_m=1.0,
                           tile_origin_xy=(0.0, 0.0), grid_shape=(10, 10),
                           robot_pose=None)


_CANDS = [
    {"fid": "prior_7", "xy": [3.0, 4.0], "class": "depot",
     "search_text": "depot"},
    {"fid": "prior_1", "xy": [10.0, 20.0], "class": "gate",
     "search_text": "gate"},
    {"fid": "prior_2", "xy": [10.0, 40.0], "class": "tower",
     "search_text": "tower"},
    {"fid": "car_1", "xy": [20.0, 20.0], "class": "car",
     "search_text": "car"},
    {"fid": "car_2", "xy": [0.0, 30.0], "class": "car",
     "search_text": "car"},
]


def _drive(script, *, fly=None, declare=None, flight_budget=8, turn_cap=48,
           transcript=None, cands=None, retask=None):
    flights: list[dict] = []

    def _fly(item):
        flights.append(item)
        if fly is not None:
            return fly(item)
        return {"stop_id": f"stop_{len(flights) - 1:03d}",
                "target": item["target_fid"], "pose_xy": [1.0, 2.0],
                "new_detections": 1, "cue": "the cue"}

    declared: list[list[dict]] = []

    def _declare(items):
        declared.append(list(items))
        if declare is not None:
            return declare(items)
        return {"declared": [f"{i['class']} ({i['role']})" for i in items]}

    fr = FakeReasoner(script)
    reply = advance_reasoner_one_round(
        reasoner=fr,
        user_message="Find the goal object.",
        ctx=_ctx(),
        rid_tag="m0",
        transcript=transcript,
        compute_namespace={"cands": list(_CANDS if cands is None else cands),
                           "anchors": [], "cell_m": 1.0},
        diag={},
        mission_spec=MissionSpec(fly=_fly, declare_concepts=_declare,
                                 flight_budget=flight_budget,
                                 turn_cap=turn_cap),
        retask=retask,
    )
    assert isinstance(reply, MissionReply)
    return reply, fr, flights, declared


def _tool_results(fr: FakeReasoner) -> list[dict]:
    out = []
    for m in fr.messages:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(m["content"]))
            except Exception:
                out.append({"raw": m["content"]})
    return out
































def test_unknown_tool_names_the_one_verb():
    _, fr, _, _ = _drive([
        [("find", {"query": "Wellington Road"})],
        [("teleport", {"x": 1})],
        "over",
    ])
    results = _tool_results(fr)
    for r in results[:2]:
        assert "unknown tool" in r["error"]
        # THE VERB IT NAMES IS THE ONE SERVED. The mission path serves
        # CODE_TOOLS ([run]); this answered "available: say" for the whole
        # life of the code arm, so the one message sent when the model has
        # already lost the tool name handed it a retired one.
        assert "available: run" in r["error"], r["error"]
        assert "say" not in r["error"], r["error"]








def test_sanitizer_is_directly_pinned():
    ok, errs = _mission_sanitize_moves(["[900, 640]", "marked_region"], {})
    assert ok == [{"target_fid": "[900, 640]"},
                  {"target_fid": "marked_region"}]
    assert errs == []
    staged = {"v1": {"handle": "v1", "from_xy": [3.0, 4.0],
                     "up_unit": [0.0, 1.0]}}
    ok, errs = _mission_sanitize_moves(["v1"], staged)
    assert ok == [{"target_fid": "[3, 4]", "via": "v1"}]
    assert errs == []
    ok, errs = _mission_sanitize_moves("not-a-list", {})
    assert ok == [] and errs == []








def test_second_consecutive_empty_turn_still_ends_the_mission():
    # The ending stays model-owned: a repeated empty is the ending, so a
    # dead serve cannot loop the mission on nudges.
    reply, fr, _, _ = _drive(["", ""])
    assert reply.ended_by == "reply"
    assert reply.reply_text == ""
    assert reply.turns_used == 2


# The wrong-channel reply is tested where its detector lives: the arm
# serves `run`, so the shape that gets dropped is a BLOCK, not a sentence.
# See test_code_surface_arm.py --
# test_a_block_written_into_the_reply_channel_is_repaired_not_an_ending,
# test_a_second_block_in_the_reply_channel_still_ends_the_mission,
# test_prose_that_narrates_a_call_is_a_real_ending.


def test_a_flight_shows_what_it_saw():
    """The harness renders the flight's view to segment it and then threw
    the pixels away, handing back a COUNT of the classes it was told to
    watch. That count is what makes "0 detections" read as absence when it
    means "nothing you asked for" -- a log study over 146 reached-then-missed
    episodes found the model trusting it while standing on the target.

    And the model paid to undo it: 45% of 4,691 flights were followed
    immediately by a look or map call asking to see where it had just gone.
    """
    import maps.search.parse_compile.reasoner_loop as rl
    src = inspect.getsource(rl)
    i = src.index("THE FLIGHT SHOWS WHAT IT SAW")
    # anchored at BOTH ends -- a fixed char count silently stopped covering
    # the block the first time a comment grew inside it
    body = src[i:src.index("mission_flights += 1", i)]
    # rendered through the SAME look renderer, so live and replay stay
    # byte-identical and no second render path can drift
    assert "look_renderer([{" in body
    # the pixels reach the model, not just the ack
    assert "pending_vision.append" in body
    # and a render failure never takes the flight down with it
    assert "never crash a flight" in body










def test_three_replies_in_a_row_with_no_flight_is_a_confident_abstain():
    """TWO repairs, then the third reply is the ending.

    Zero flights stays an honest zero and the harness never picks a target:
    "the commit is the LAST flight's target" forbids inventing one where
    nothing flew, and asking again is not inventing. What it does forbid is
    asking forever, so the streak ends it -- a model that says nothing three
    times running has abstained, and that is a legal answer.
    """
    reply, fr, flights, _ = _drive(
        ["nothing here.", "nothing here.", "nothing here."])
    assert reply.ended_by == "reply"
    assert reply.flights_flown == 0
    assert flights == []
    nudges = [m for m in fr.messages
              if m.get("role") == "user" and "flown nowhere" in
              str(m.get("content"))]
    assert len(nudges) == 2, nudges






# --------------------------------------------------------------------------
# The operator's channel into a flying mission (the field console's re-task).
# The bench passes no channel and must stay one-user-message; the console
# passes one and its text has to reach the model as a plain user turn.
# --------------------------------------------------------------------------


def _plain_user_turns(fr: FakeReasoner) -> list[str]:
    """User-role turns that are neither a harness repair nor a crop legend.

    Repairs are prefixed by ``append_repair`` and vision rides
    ``append_vision_user``; neither is the operator speaking.
    """
    return [str(m.get("content")) for m in fr.messages
            if m.get("role") == "user"
            and not str(m.get("content")).startswith("[Repair] ")]










