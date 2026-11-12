"""The mission contract: a block flies on the call; the plain reply ends it."""

from __future__ import annotations


from maps.mission.tests.fakes import drive, run, tool_results


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


def test_a_quoted_handle_moves_as_the_set_it_names():
    cands = [{"fid": "detected_0", "xy": [0.0, 0.0], "class": "car",
              "search_text": "car"},
             {"fid": "detected_1", "xy": [10.0, 0.0], "class": "car",
              "search_text": "car"}]
    _, fr, flown, _ = drive(
        [[run('r1 = find("car")\ngo_to("r1")\n')],
         [run('r2 = find("zebra crossing")\nexplore("r2")\n')]]
        + ["done."] * 3, cands=cands)
    went, empty = (r["error"] for r in tool_results(fr)[:2])
    assert flown == []
    # The handle's own name and the remedy, never an internal ref.
    assert "r1 is a set of 2 row(s)" in went and "go_to(r1[0])" in went, went
    assert "r2 holds 0 row(s) here" in empty, empty
    assert "fid-bearing" not in empty, empty


def test_second_consecutive_empty_turn_still_ends_the_mission():
    # The ending stays model-owned: a repeated empty is the ending, so a
    # dead serve cannot loop the mission on nudges.
    reply, fr, _, _ = drive(["", ""])
    assert reply.ended_by == "reply"
    assert reply.turns_used == 2


# The wrong-channel reply is tested where its detector lives: the arm
# serves `run`, so the shape that gets dropped is a BLOCK, not a sentence.
# See maps/library/tests/test_library_arm.py --
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


def test_a_followup_turn_keeps_the_missions_names_stops_and_watchlist():
    """An operator turn after a report is an UPDATE to one mission: what the
    first turn bound, flew and watched is still there."""
    _, fr, _, declared = drive(
        [[run('cars = find("car")\nlook_for("car")\ngo_to("prior_7")')], "report"])
    n = len(tool_results(fr))
    drive([[run("print(len(cars))")],
           [run('stand(at="stop_000", facing="north")')], "report"], reasoner=fr)
    named, stood = tool_results(fr)[n:]
    assert "error" not in named and named["watching"] == ["car (0 detected)"], named
    assert stood["stance"]["staged"][0]["at_xy"] == [1.0, 2.0], stood
    assert declared == [["car"]]


def test_a_followup_after_a_turn_that_flew_nowhere_is_still_preflight():
    _, fr, _, _ = drive(["nothing."] * 3, preflight=True)
    drive(["still nothing."] * 3, reasoner=fr)
    assert fr.mission[0].surface.preflight
    drive([[run('go_to("prior_7")')], "report"], reasoner=fr)
    assert not fr.mission[0].surface.preflight


def test_look_and_its_marks_take_every_point_a_stance_takes():
    """A flown stop id and a staged vN handle are places stand(at=) and
    show_map(around=) take; look takes them too, as a subject and as a mark."""
    from maps.mission.tests.fakes import jpeg

    asked = []

    def renderer(reqs):
        asked.extend(reqs)
        return [{"jpeg": jpeg(), "center_fid": r["center_fid"], "legend": []} for r in reqs]

    drive([[run('go_to("prior_7")')],
           [run('stand(at="stop_000", facing="north")')],
           [run('look("stop_000")\nlook("v1", marks=["stop_000"])')],
           "done"], look_renderer=renderer, obs_minor_m=None)
    looks = [r for r in asked if r["center_fid"] in ("stop_000", "v1")]
    assert [(r["center_fid"], r["center_xy"]) for r in looks] == [
        ("stop_000", [1.0, 2.0]), ("v1", [1.0, 2.0])], asked
    assert [m["fid"] for m in looks[1]["marks"]] == ["stop_000", "v1"], looks[1]


_TIE_CANDS = [
    {"fid": "prior_1", "xy": [0.0, 0.0], "class": "depot", "search_text": "depot"},
    {"fid": "prior_2", "xy": [0.0, 5.0], "class": "tower", "search_text": "tower"},
    {"fid": "car_1", "xy": [10.0, 0.0], "class": "car", "search_text": "car"},
    {"fid": "car_2", "xy": [-10.0, 0.0], "class": "car", "search_text": "car"},
]


def _tie_note(cands):
    _, fr, _, _ = drive([[run('r = find("car", nearest=["depot", "tower"])\nr')],
                         "done"], cands=cands)
    return tool_results(fr)[0].get("note")


def test_the_tie_note_reads_every_ordering_key():
    """Rows tied on the first key but separated by the second are no tie;
    rows tied on every key are, and the note names every key."""
    moved = [dict(c, xy=[10.0, 5.0]) if c["fid"] == "prior_2" else c
             for c in _TIE_CANDS]
    assert _tie_note(moved) is None
    assert _tie_note(_TIE_CANDS) == (
        "2 of 2 rows tie on every ordering column ('distance to depot', "
        "'distance to tower') -- this ordering does not separate them, so "
        "its rank 0 is arbitrary among the tie")


def test_a_retask_on_the_turn_cap_turn_stays_queued():
    """The turn cap ends the mission before the re-task queue drains, so a
    re-task arriving then is still queued, never appended to a thread that
    is over."""
    queue: list = []

    def fly(item):
        queue.append("operator: also check the gate")
        return {"stop_id": "stop_000", "target": item["target_fid"],
                "pose_xy": [1.0, 2.0], "new_detections": 0}

    reply, fr, _, _ = drive([[run('go_to("prior_7")')]], turn_cap=1, fly=fly,
                            retask=lambda: queue.pop(0) if queue else None)
    assert reply.ended_by == "turn_cap"
    assert queue == ["operator: also check the gate"]
    assert not [m for m in fr.messages if "also check the gate" in str(m.get("content"))]


def test_a_pass_that_flew_nothing_leaves_its_segment_uncovered():
    """A segment counts as covered once a pass over it flies: after a
    pass that flew nothing, the next pass's note still names it."""
    cands = [{"fid": f"road_{i}", "xy": [10.0 * i, 0.0], "class": "road",
              "name": "Main St", "search_text": "main st road"} for i in (1, 2)]
    tries: list = []

    def fly(item):
        tries.append(item)
        if len(tries) == 1:
            return {"error": "the aircraft refused that stop"}
        return {"stop_id": f"stop_{len(tries):03d}", "target": item["target_fid"],
                "pose_xy": [1.0, 2.0], "new_detections": 0}

    _, fr, _, _ = drive([[run('explore("road_1")')], [run('explore("road_2")')],
                         "done"], cands=cands, fly=fly)
    failed, flown = tool_results(fr)[:2]
    assert "error" in failed, failed
    assert "'Main St' has 1 more segment(s) no pass has covered (road_1)" in \
        flown["explored"]["note"], flown


def test_a_turn_the_output_cap_cut_off_spends_no_repair():
    # A truncated empty turn is the serve's cut, nudged on its own: the one
    # repair is still there for a real dropped reply after it.
    cut = {"content": "", "finish_reason": "length"}
    reply, fr, flights, _ = drive([cut, "", [run('go_to(prior_7)\n')], "done"])
    assert reply.ended_by == "reply" and reply.turns_used == 4, reply
    assert len(flights) == 1


def test_three_cut_off_turns_in_a_row_each_get_their_own_nudge():
    # Up to 3 in a row are free, so the repair still covers the empty reply.
    cut = {"content": "", "finish_reason": "length"}
    reply, fr, flights, _ = drive([cut, cut, cut, "", [run('go_to(prior_7)\n')], "done"])
    assert reply.ended_by == "reply" and reply.turns_used == 6, reply
    assert len(flights) == 1


def test_repeated_cut_off_turns_still_end_the_mission():
    # A serve that only ever truncates cannot loop the mission on nudges.
    cut = {"content": "", "finish_reason": "length"}
    # Three nudges, the fourth cut spends the repair, the fifth ends it.
    reply, fr, _, _ = drive([cut] * 5)
    assert reply.ended_by == "reply" and reply.turns_used == 5, reply
