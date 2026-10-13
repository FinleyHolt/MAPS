"""The mission contract: a block flies on the call; the plain reply ends it."""

from __future__ import annotations


from maps.search.tests.fakes import (
    FakeReasoner,
    bare_ctx,
    drive,
    run,
    tool_results,
    unarmed_boost,
)


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


def test_move_refuses_a_result_handle_and_names_what_it_takes():
    from maps.search.parse_compile.reasoner_loop import MissionSpec, _Mission
    flown = []
    m = _Mission(reasoner=FakeReasoner([]), ctx=bare_ctx(),
                 compute_namespace={"cands": [], "anchors": [], "cell_m": 1.0},
                 mission_spec=MissionSpec(fly=flown.append,
                                          declare_concepts=lambda _i: None),
                 boost_queue=unarmed_boost)
    out = m._move_to("r3[0:2]")
    assert flown == []
    rej = out["flown"]["rejected"]
    assert out["flown"]["flown"] == [] and rej[0]["index"] == 0, out
    assert "is a result handle" in rej[0]["error"], out
    # what move accepts, and only that
    assert "staged" not in out["error"] and "flown stop id" in out["error"], out


def test_second_consecutive_empty_turn_still_ends_the_mission():
    # The ending stays model-owned: a repeated empty is the ending, so a
    # dead serve cannot loop the mission on nudges.
    reply, fr, _, _ = drive(["", ""])
    assert reply.ended_by == "reply"
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
    assert len(flights) == 1
    assert "error" not in tool_results(fr)[0], tool_results(fr)[0]
    assert fr.vision == []


def test_three_replies_in_a_row_with_no_flight_is_a_confident_abstain():
    """TWO repairs, then the third reply is the ending."""
    reply, fr, flights, _ = drive(
        ["nothing here.", "nothing here.", "nothing here."])
    assert reply.ended_by == "reply"
    assert flights == []
    nudges = [m for m in fr.messages
              if m.get("role") == "user" and "flown nowhere" in
              str(m.get("content"))]
    assert len(nudges) == 2, nudges


def test_a_reply_after_a_retask_answers_with_the_missions_last_flight():
    """A re-task turn that replies without flying still holds the earlier
    turn's flight: no zero-flight nudge, no abstain."""
    _, fr, flights, _ = drive([[run("go_to(prior_7)\n")], "the depot."])
    assert len(flights) == 1
    reply, fr, flights, _ = drive(["the north side holds nothing new."],
                                  reasoner=fr)
    assert reply.ended_by == "reply"
    assert reply.turns_used == 1
    assert not [m for m in fr.messages if "flown nowhere" in
                str(m.get("content"))]
