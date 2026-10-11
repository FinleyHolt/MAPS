"""The mission contract: a block flies on the call; the plain reply ends it."""

from __future__ import annotations


from maps.search.parse_compile.reasoner_loop import (
    _mission_sanitize_moves,
)
from maps.search.tests.fakes import drive, run, tool_results


def test_unknown_tool_names_the_one_verb():
    _, fr, _, _ = drive([
        [("find", {"query": "Wellington Road"})],
        [("teleport", {"x": 1})],
        "over",
    ])
    results = tool_results(fr)
    for r in results[:2]:
        assert "unknown tool" in r["error"]
        # THE VERB IT NAMES IS THE ONE SERVED.
        assert "available: run" in r["error"], r["error"]
        assert "say" not in r["error"], r["error"]


def test_unparseable_arguments_serve_the_parse_error():
    _, fr, _, _ = drive([
        [("run", {"__args_parse_error": "Expecting ',' delimiter at pos 9"})],
        "over",
    ])
    err = tool_results(fr)[0]["error"]
    assert "Expecting ',' delimiter at pos 9" in err and "nothing ran" in err, err


def test_sanitizer_is_directly_pinned():
    ok, errs = _mission_sanitize_moves(["[900, 640]", "marked_region"])
    assert ok == [{"target_fid": "[900, 640]"},
                  {"target_fid": "marked_region"}]
    assert errs == []
    ok, errs = _mission_sanitize_moves(["r3[0:2]"])
    assert ok == [] and "is a result handle" in errs[0]["error"], errs
    # what move accepts, and only that
    assert "staged" not in errs[0]["error"] and "flown stop id" in errs[0]["error"], errs


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


def test_a_flight_view_that_fails_to_render_never_crashes_the_flight():
    def _broken(reqs):
        raise RuntimeError("renderer down")

    reply, fr, flights, _ = drive([[run("go_to(prior_7)\n")], "done"],
                                  look_renderer=_broken)
    assert len(flights) == 1 and reply.flights_flown == 1
    assert "error" not in tool_results(fr)[0], tool_results(fr)[0]
    assert fr.vision == []


def test_three_replies_in_a_row_with_no_flight_is_a_confident_abstain():
    """TWO repairs, then the third reply is the ending."""
    reply, fr, flights, _ = drive(
        ["nothing here.", "nothing here.", "nothing here."])
    assert reply.ended_by == "reply"
    assert reply.flights_flown == 0
    assert flights == []
    nudges = [m for m in fr.messages
              if m.get("role") == "user" and "flown nowhere" in
              str(m.get("content"))]
    assert len(nudges) == 2, nudges
