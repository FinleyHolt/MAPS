"""The mission contract: a block flies on the call; the plain reply ends it.

Drives ``advance_reasoner_one_round(mission_spec=...)`` with a scripted
reasoner — no serve, no SAM. This file owns the harness the surface's other
test files drive through (``FakeReasoner``, ``bare_ctx``, ``CANDS``, ``drive``,
``tool_results``) and pins the loop-level semantics: the ONE model-owned
ending is a clean no-tool-call reply, a reply with nothing flown behind it is
repaired rather than accepted, an unknown tool name is answered and never
executed, and the liveness bounds behave with no budget fact ever served.

The library's own semantics — what each call does to the world and what comes
back — are pinned in ``test_code_surface_arm`` and ``test_python_surface``.
"""

from __future__ import annotations

import inspect

from maps.search.parse_compile.reasoner_loop import (
    _mission_sanitize_moves,
)
from maps.search.tests.fakes import FakeReasoner, drive, tool_results


def test_unknown_tool_names_the_one_verb():
    _, fr, _, _ = drive([
        [("find", {"query": "Wellington Road"})],
        [("teleport", {"x": 1})],
        "over",
    ])
    results = tool_results(fr)
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
    reply, fr, _, _ = drive(["", ""])
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
    reply, fr, flights, _ = drive(
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










