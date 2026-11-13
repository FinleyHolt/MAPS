"""The arming channel's receipts tell the truth."""

from __future__ import annotations

import json

import pytest

from maps.mission.tests.fakes import drive, episode_world, run, tool_results
from maps.mission.watch_verbs import _named_feature_note


def test_named_feature_note_fires_on_a_world_state_name():
    note = _named_feature_note(["airway boulevard"], {"airway boulevard"})
    assert "NAMED feature" in note
    assert "airway boulevard" in note
    # the arm stands as said -- the note must say so, never imply a rewrite
    assert "Watched as said" in note


def test_named_feature_note_is_silent_on_visual_classes():
    assert _named_feature_note(["house", "pool"], {"airway boulevard", "lime avenue"}) is None
    assert _named_feature_note([], {"airway boulevard"}) is None
    assert _named_feature_note(["house"], set()) is None


def test_named_feature_note_is_case_blind():
    assert _named_feature_note(["Airway Boulevard"], {"airway boulevard"}) is not None


# --------------------------------------------------------------------------- #
# declare_mission_concepts: the setter failure reaches the receipt
# --------------------------------------------------------------------------- #


def test_setter_failure_reaches_the_receipt():
    def _boom(_cs):
        raise RuntimeError("sam server gone")

    rcpt = episode_world(sam_setter=_boom).declare_mission_concepts(["house"])
    assert "sam server gone" in rcpt


def test_clean_setter_keeps_a_clean_receipt():
    assert episode_world(sam_setter=lambda cs: None).declare_mission_concepts(["house"]) is None


def test_the_placed_form_also_arms_the_standing_watchlist():
    boosts = []

    def _boost(concepts, where):
        boosts.append((concepts, where))
        return {"queued": True}

    _, fr, _, declared = drive(
        [[run('look_for("pool", at="the depot")\n')], "done"], boost_queue=_boost
    )
    assert [c for c, _w in boosts] == [["pool"]]
    assert "pool" in declared[-1], declared
    ack = json.dumps(tool_results(fr)[0])
    assert "standing watchlist for every later flight" in ack, ack


def test_a_retro_boost_at_a_flown_stop_also_arms_the_standing_watchlist():
    _, fr, _, declared = drive(
        [[run('look_for("pool", at="stop_000")\n')], "done"],
        boost_queue=lambda concepts, where: {"retro": [{"n_dets": 0}]},
    )
    assert "pool" in declared[-1], declared
    ack = json.dumps(tool_results(fr)[0])
    assert "pool (target)" in ack and "boost queued" not in ack, ack


def test_the_declared_union_is_canonical_and_keeps_first_seen_order():
    pushed = []
    episode_world(sam_setter=pushed.append).declare_mission_concepts(
        ["Red  Car", "tree", "None", "red car", None]
    )
    assert pushed[0] == ["red car", "tree"]


def test_a_quoted_point_boost_re_segments_the_flown_stop_it_falls_on():
    from maps.perception.watchlist import SegmenterTasking

    d = SegmenterTasking(retro_perceiver=lambda flown, cs: {"n_dets": 1, "stop": flown["stop_id"]})
    d.record_flown({"pose_xy": (100.0, 200.0), "stop_id": "stop_000", "target_fids": ["prior_1"]})
    ack = d.boost_queue(["pool"], ["[110, 205]"])
    assert ack["retro"] == [{"where": [110.0, 205.0], "n_dets": 1, "stop": "stop_000"}], ack
    assert "queued" not in ack


def test_a_boost_at_an_unflown_stop_names_it_and_queues_nothing():
    from maps.perception.watchlist import SegmenterTasking

    assert SegmenterTasking().boost_queue(["car"], ["stop_009"]) == {
        "error": "no boost target resolved; nothing was queued or re-segmented",
        "unknown_where": ["stop_009"],
    }


def test_a_queued_boost_rides_every_flight_until_its_target_flies():
    from maps.perception.sam.detect_client import SamDetectClient

    prov = SamDetectClient()
    loop = episode_world(sam_setter=prov.set_concepts)
    loop.declare_mission_concepts(["car"])
    assert loop.boost_queue(["pool"], ["prior_1"])["queued"] == ["pool"]
    for target in ("prior_2", "prior_1"):  # the boost rides until prior_1 flies
        loop.start_flight()
        assert prov.concepts == ["car", "pool"]
        loop.record_flown_stop({"pose_xy": (0.0, 0.0), "stop_id": "s", "target_fids": [target]})
        assert prov.concepts == ["car"]
    loop.start_flight()
    assert prov.concepts == ["car"]


def test_the_segmenter_keeps_the_declared_order_across_a_flight():
    """A re-declared class moves to the end of the declared union, and every
    later push (a flight's start, its recorded stop) sends that same order."""
    pushed = []
    loop = episode_world(sam_setter=pushed.append)
    loop.declare_mission_concepts(["car", "house"])
    loop.declare_mission_concepts(["house", "car"])
    loop.boost_queue(["pool"], ["prior_1"])
    loop.start_flight()
    loop.record_flown_stop({"pose_xy": (0.0, 0.0), "stop_id": "s", "target_fids": ["prior_9"]})
    assert pushed == [
        ["car", "house"],
        ["house", "car"],
        ["house", "car", "pool"],
        ["house", "car"],
    ]


def test_an_empty_watchlist_is_told_once_per_run_not_per_flight():
    def _fly(item):
        return {
            "stop_id": "stop_000",
            "target": item["target_fid"],
            "pose_xy": [1.0, 2.0],
            "new_detections": 0,
            "cue": "the cue",
        }

    _, fr, flights, _ = drive([[run('go_to("prior_7")\ngo_to("prior_1")\n')], "done"], fly=_fly)
    assert len(flights) == 2, flights
    served = json.dumps(tool_results(fr)[0])
    assert served.count('look_for(\\"<class>\\")') == 1, served


# --------------------------------------------------------------------------- #
# look_for's served limits: up to four classes, each a THING, never a colour
# --------------------------------------------------------------------------- #


def test_a_fifth_class_is_refused_and_the_four_stay_armed():
    _, fr, _, declared = drive(
        [
            [run('look_for("car", "house", "pool", "tree", "shed")\n')],
            [run('look_for("car", "house", "pool", "tree")\n')],
            [run('look_for("car", "car")\n')],
            [run('look_for("shed")\n')],
            "done",
        ]
    )
    out = tool_results(fr)
    assert "up to four classes" in out[0]["error"], out[0]
    assert "error" not in out[1] and "error" not in out[2], out[1:3]
    assert "up to four classes" in out[3]["error"], out[3]
    assert declared == [["car", "house", "pool", "tree"], ["house", "pool", "tree", "car"]]


def test_a_placed_fifth_class_queues_no_boost():
    boosts = []
    _, fr, _, declared = drive(
        [
            [run('look_for("car", "house", "pool", "tree")\n')],
            [run('look_for("shed", at="the depot")\n')],
            "done",
        ],
        boost_queue=lambda concepts, where: boosts.append(concepts) or {"queued": True},
    )
    assert "up to four classes" in tool_results(fr)[1]["error"]
    assert boosts == [] and declared == [["car", "house", "pool", "tree"]], (boosts, declared)


def _named(*names):
    from types import SimpleNamespace

    return SimpleNamespace(
        nodes={
            f"prior_{i}": SimpleNamespace(name=n, provenance="prebaked")
            for i, n in enumerate(names)
        }
    )


@pytest.mark.parametrize(
    "said, rule",
    [
        ('look_for("red car")', 'never a colour: "red car" carries the colour "red"'),
        ('look_for("car", "Grey van")', 'never a colour: "Grey van" carries the colour "Grey"'),
        ('look_for("First Church")', 'no proper nouns: "First Church" is a proper noun'),
        ('look_for("Green Street")', 'no proper nouns: "Green Street" is a proper noun'),
        ('look_for("pool", at="the depot")', None),
    ],
)
def test_a_colour_or_a_proper_noun_is_refused_and_arms_nothing(said, rule):
    _, fr, _, declared = drive(
        [[run(said + "\n")], "done"],
        boost_queue=lambda concepts, where: {"queued": True},
        # the surface rebuilds from the graph on an accepted boost
        graph=_named("First Church", "Green Street") if rule else None,
    )
    out = tool_results(fr)[0]
    if rule is None:
        assert "error" not in out, out
        return
    assert out["error"].startswith("line 1: look_for takes plain visual classes"), out
    assert rule in out["error"], out
    assert declared == [], declared


@pytest.mark.parametrize("said", ["Swimming Pool", "Parking_Lot", "car"])
def test_a_title_case_class_no_feature_is_named_is_watched(said):
    """A proper noun is a world-state name written as one; a Title-Case
    common class, or a lowercase class a feature happens to be named, is
    watched as said."""
    _, fr, _, declared = drive(
        [[run(f'look_for("{said}")\n')], "done"], graph=_named("car", "First Church")
    )
    assert "error" not in tool_results(fr)[0], tool_results(fr)[0]
    assert declared == [[said.lower().replace("_", " ")]], declared


def test_the_cap_refusal_names_its_line():
    _, fr, _, _ = drive(
        [[run('look_for("car", "house")\nlook_for("pool", "tree", "shed")\n')], "done"]
    )
    assert tool_results(fr)[0]["error"].startswith("line 2: look_for watches up to four"), (
        tool_results(fr)[0]
    )


def test_a_capitalised_first_word_is_a_class():
    _, fr, _, declared = drive([[run('look_for("Car")\n')], "done"])
    assert "error" not in tool_results(fr)[0]
    assert declared == [["car"]], declared
