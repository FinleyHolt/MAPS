"""The arming channel's receipts tell the truth."""

from __future__ import annotations

import json
import threading


from maps.search.parse_compile.reasoner_loop import _named_feature_note
from maps.search.tests.fakes import drive, run, tool_results


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


def _duck(setter):
    from maps.search.parse_compile.perception_director import (
        PerceptionDirector,
    )

    director = PerceptionDirector(push=lambda cs: None)
    director.init_watchlist([])

    class _S:
        perception_director = director

    class _Duck:
        state = _S()
        _lock = threading.Lock()
        _sam_concept_setter = staticmethod(setter)

    return _Duck()


def test_setter_failure_reaches_the_receipt():
    from maps.search.search_loop import SearchLoop

    def _boom(_cs):
        raise RuntimeError("sam server gone")

    rcpt = SearchLoop.declare_mission_concepts(
        _duck(_boom), [{"class": "house", "role": "target"}])
    assert "sam server gone" in rcpt["error"]
    assert rcpt["declared"] == ["house (target)"]      # what DID happen


def test_clean_setter_keeps_a_clean_receipt():
    from maps.search.search_loop import SearchLoop

    rcpt = SearchLoop.declare_mission_concepts(
        _duck(lambda cs: None), [{"class": "house", "role": "target"}])
    assert "error" not in rcpt


def test_the_placed_form_also_arms_the_standing_watchlist():
    boosts = []

    def _boost(concepts, where):
        boosts.append((concepts, where))
        return {"queued": True}

    _, fr, _, declared = drive([[run('look_for("pool", at="the depot")\n')], "done"],
                               boost_queue=_boost)
    assert [c for c, _w in boosts] == [["pool"]]
    assert {"class": "pool", "role": "target"} in declared[-1], declared
    ack = json.dumps(tool_results(fr)[0])
    assert "standing watchlist for every later flight" in ack, ack


def test_concept_union_keeps_first_seen_order():
    from maps.search.parse_compile.compiler import (
        _extract_reasoner_concepts,
        concept_union,
    )
    specs = _extract_reasoner_concepts({"sam_concepts": [
        {"class": "Red  Car", "role": "target"}, "tree", "None",
        {"class": "red car", "role": "anchor"},
        {"class": "tree", "role": "bogus"},
    ]})
    assert concept_union(specs) == ["red car", "tree"]


def test_a_quoted_point_boost_re_segments_the_flown_stop_it_falls_on():
    from maps.search.parse_compile.perception_director import PerceptionDirector

    d = PerceptionDirector(retro_perceiver=lambda flown, cs: {"n_dets": 1, "stop": flown["stop_id"]})
    d.retire_after_round([{"pose_xy": (100.0, 200.0), "stop_id": "stop_000", "target_fids": ["prior_1"]}])
    ack = d.boost_queue(["pool"], ["[110, 205]"])
    assert ack["retro"] == [{"where": [110.0, 205.0], "n_dets": 1, "stop": "stop_000"}], ack
    assert "queued" not in ack
