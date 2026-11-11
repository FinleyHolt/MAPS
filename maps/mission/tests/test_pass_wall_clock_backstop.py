"""A pass stops flying on the clock, because the thing that kills it is a clock."""

from __future__ import annotations

import re
import time

from maps.mission.tests.fakes import explore_drive, run


def test_a_pass_already_past_its_deadline_flies_nothing():
    _reply, out, flights, _asked, _rend = explore_drive(
        [[run('explore("prior_7")\n')], "done"],
        deadline_epoch=time.time() - 60.0,
    )
    assert flights == [], flights
    assert "could be flown" in str(out[0].get("error") or ""), out[0]
    # the model never hears a budget fact or a clock
    blob = repr(out[0])
    for word in ("deadline", "budget", "wall-clock", "wall clock",
                 "seconds left", "time left", "flights remaining"):
        assert word not in blob.lower(), (word, blob)
    assert not re.search(r"1[6-9]\d{8}", blob), blob


def test_a_pass_cut_midway_flies_what_it_had_and_reads_as_truncated(monkeypatch):
    # A clock the stops advance, not wall time.
    clock = {"now": 1_000_000.0}

    def _advance(_item):
        clock["now"] += 100.0

    import maps.mission.loop as rl

    monkeypatch.setattr(rl.time, "time", lambda: clock["now"])
    _reply, out, flights, _asked, _rend = explore_drive(
        [[run('explore("prior_7")\n')], "done"],
        deadline_epoch=clock["now"] + 150.0,
        on_fly=_advance,
    )

    # stop 1 at t=1_000_000 (under), stop 2 at t=1_000_100 (under),
    # stop 3 at t=1_000_200 (over) -> two flown of three.
    assert len(flights) == 2, flights
    ack = out[0]["explored"]
    assert ack["stops"] == 2, ack
    note = str(ack.get("note") or "")
    assert "ended after 2 of its stops" in note, note
    # and it did NOT keep claiming the whole walk
    assert "covered 200 m" not in note, note


def test_the_flight_arm_still_fires_on_its_own():
    _reply, out, flights, _asked, _rend = explore_drive(
        [[run('explore("prior_7")\n')], "done"],
        stops=[[float(x), 200.0] for x in range(100, 100 + 40 * 5, 5)],
    )
    # flight_budget is 32 in this fixture; the walk asks for 40 stops.
    assert len(flights) == 32, len(flights)
    assert out[0]["explored"]["stops"] == 32
