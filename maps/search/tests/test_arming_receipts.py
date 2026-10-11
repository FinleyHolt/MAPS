"""The arming channel's receipts tell the truth (acquisition campaign).

Three lies measured or latent on f40088a:calcity_full_replyrepair:
the boost receipt never said the standing watchlist stayed empty (40 of
the facing family's 49 never-armed episodes used ONLY the boost form);
a proper noun reaching the segmenter as a class drew no comment (3
world-state names among 18 watch classes); and a failing segmenter
setter produced a clean ``declared`` receipt (zero incidents this run,
but the swallow existed).
"""

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
