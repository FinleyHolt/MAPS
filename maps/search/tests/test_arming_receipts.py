"""The arming channel's receipts tell the truth."""

from __future__ import annotations

import json

from maps.search.parse_compile.reasoner_loop import _named_feature_note
from maps.search.tests.fakes import drive, run, search_loop, tool_results


def test_named_feature_note_fires_on_a_world_state_name():
    note = _named_feature_note(["airway boulevard"], {"airway boulevard"})
    assert "NAMED feature" in note
    assert "airway boulevard" in note
    # the arm stands as said -- the note must say so, never imply a rewrite
    assert "Watched as said" in note


def test_named_feature_note_is_silent_on_visual_classes():
    assert _named_feature_note(["house", "pool"],
                               {"airway boulevard", "lime avenue"}) is None
    assert _named_feature_note([], {"airway boulevard"}) is None
    assert _named_feature_note(["house"], set()) is None


def test_named_feature_note_is_case_blind():
    assert _named_feature_note(["Airway Boulevard"],
                               {"airway boulevard"}) is not None


# --------------------------------------------------------------------------- #
# declare_mission_concepts: the setter failure reaches the receipt
# --------------------------------------------------------------------------- #


def test_setter_failure_reaches_the_receipt():
    def _boom(_cs):
        raise RuntimeError("sam server gone")

    rcpt = search_loop(sam_setter=_boom).declare_mission_concepts(
        ["house"])
    assert "sam server gone" in rcpt


def test_clean_setter_keeps_a_clean_receipt():
    assert search_loop(sam_setter=lambda cs: None).declare_mission_concepts(
        ["house"]) is None


def test_the_placed_form_also_arms_the_standing_watchlist():
    boosts = []

    def _boost(concepts, where):
        boosts.append((concepts, where))
        return {"queued": True}

    _, fr, _, declared = drive([[run('look_for("pool", at="the depot")\n')], "done"],
                               boost_queue=_boost)
    assert [c for c, _w in boosts] == [["pool"]]
    assert "pool" in declared[-1], declared
    ack = json.dumps(tool_results(fr)[0])
    assert "standing watchlist for every later flight" in ack, ack


def test_a_retro_boost_at_a_flown_stop_also_arms_the_standing_watchlist():
    _, fr, _, declared = drive(
        [[run('look_for("pool", at="stop_000")\n')], "done"],
        boost_queue=lambda concepts, where: {"retro": [{"n_dets": 0}]})
    assert "pool" in declared[-1], declared
    ack = json.dumps(tool_results(fr)[0])
    assert "pool (target)" in ack and "boost queued" not in ack, ack


def test_the_declared_union_is_canonical_and_keeps_first_seen_order():
    pushed = []
    search_loop(sam_setter=pushed.append).declare_mission_concepts(
        ["Red  Car", "tree", "None", "red car", None])
    assert pushed[0] == ["red car", "tree"]


def test_a_quoted_point_boost_re_segments_the_flown_stop_it_falls_on():
    from maps.search.parse_compile.perception_director import PerceptionDirector

    d = PerceptionDirector(retro_perceiver=lambda flown, cs: {"n_dets": 1, "stop": flown["stop_id"]})
    d.record_flown({"pose_xy": (100.0, 200.0), "stop_id": "stop_000", "target_fids": ["prior_1"]})
    ack = d.boost_queue(["pool"], ["[110, 205]"])
    assert ack["retro"] == [{"where": [110.0, 205.0], "n_dets": 1, "stop": "stop_000"}], ack
    assert "queued" not in ack


def test_a_boost_at_an_unflown_stop_names_it_and_queues_nothing():
    from maps.search.parse_compile.perception_director import PerceptionDirector

    assert PerceptionDirector().boost_queue(["car"], ["stop_009"]) == {
        "error": "no boost target resolved; nothing was queued or re-segmented",
        "unknown_where": ["stop_009"]}


def test_a_queued_boost_segments_its_flights_in_watchlist_order_until_its_target_flies():
    from maps.perception.sam.concept_provider import ReasonerUnionConceptProvider

    prov = ReasonerUnionConceptProvider()
    loop = search_loop(sam_setter=prov.set_concepts)
    car, pool = "car", "pool"
    loop.declare_mission_concepts([car, pool])
    assert loop.boost_queue(["pool"], ["prior_1"])["queued"] == ["pool"]
    loop.declare_mission_concepts([pool, car])  # a re-declaration reorders the union
    assert prov.concepts == ["pool", "car"]
    for target in ("prior_2", "prior_1"):  # the boost rides until prior_1 flies
        loop.start_flight()
        assert prov.concepts == ["car", "pool"]
        loop.record_flown_stop({"pose_xy": (0.0, 0.0), "stop_id": "s", "target_fids": [target]})
        loop.declare_mission_concepts([car, pool])
        loop.declare_mission_concepts([pool, car])
    loop.start_flight()
    assert prov.concepts == ["pool", "car"]
