"""A pass stops flying on the clock, because the thing that kills it is a clock.

``_fly_moves`` bounded itself in STOPS (``flight_budget``) while the bench
harness bounds the episode in SECONDS. Those do not convert: per-stop cost
moves with tile density, watchlist width and the serving plane's health, so
the 1000-flight backstop is ~11 h of wall at 40 s/stop and minutes at 2 s.
calcity::6#6 and calcity::250#248 flew 262 and 264 stops of ONE ``explore``,
took the harness's 4 h SIGKILL, and were the only two episodes of 209 to
record no prediction row at all -- while the flight backstop sat ~738 flights
away from firing.

Stopping the FLYING is the whole fix: the model keeps its remaining turns and
ends the mission with its own commit, so the episode is SCORED instead of
lost. Nothing about the clock is served -- the model reads the same
``"not flown."`` the flight arm writes.
"""

from __future__ import annotations

import re
import time

from maps.search.tests.test_code_surface_arm import (
    _STOPS,
    _explore_drive,
    _run,
)


def test_a_pass_already_past_its_deadline_flies_nothing():
    """The arm fires on entry, not after a stop is spent.

    A pass where NOTHING flew resolves to the refusal the stop bound
    already produced for the same situation -- the model is told the pass
    did not happen and gets its turn straight back, which is the whole
    point: a turn back is a commit, a SIGKILL is a MISSING row.
    """
    _reply, out, flights, _asked, _rend = _explore_drive(
        [[_run('explore("prior_7")\n')], "done"],
        deadline_epoch=time.time() - 60.0,
    )
    assert flights == [], flights
    assert "could be flown" in str(out[0].get("error") or ""), out[0]


def test_no_deadline_flies_the_whole_pass():
    """``deadline_epoch=None`` is the FIELD path and must be untouched.

    A flight mission has no harness cap: it flies for as long
    as the crew is flying, and a timer that stopped it would be a bug with a
    aircraft in the air. ``None`` is the default for exactly that reason.
    """
    _reply, out, flights, _asked, _rend = _explore_drive(
        [[_run('explore("prior_7")\n')], "done"],
        deadline_epoch=None,
    )
    assert len(flights) == len(_STOPS), flights
    assert out[0]["explored"]["stops"] == len(_STOPS)


def test_a_pass_cut_midway_flies_what_it_had_and_reads_as_truncated():
    """The stops before the deadline are real; the note says the rest are not.

    ``_cov_note`` is built from the stops the pass PLANNED, so a cut pass
    would otherwise report the whole walk's reach and read as "that is all
    there was" -- the defect ``_cover_items`` already names for its own stop
    bound.
    """
    # A clock the stops advance, not wall time: crossing a deadline mid-pass
    # any other way means sleeping in a unit test.
    clock = {"now": 1_000_000.0}
    real_time = time.time

    def _advance(_item):
        clock["now"] += 100.0

    import maps.search.parse_compile.reasoner_loop as rl

    orig = rl.time.time
    rl.time.time = lambda: clock["now"]
    try:
        _reply, out, flights, _asked, _rend = _explore_drive(
            [[_run('explore("prior_7")\n')], "done"],
            deadline_epoch=clock["now"] + 150.0,
            on_fly=_advance,
        )
    finally:
        rl.time.time = orig
    assert rl.time.time is orig
    assert real_time() > 0

    # stop 1 at t=1_000_000 (under), stop 2 at t=1_000_100 (under),
    # stop 3 at t=1_000_200 (over) -> two flown of three.
    assert len(flights) == 2, flights
    ack = out[0]["explored"]
    assert ack["stops"] == 2, ack
    note = str(ack.get("note") or "")
    assert "ended after 2 of its stops" in note, note
    # and it did NOT keep claiming the whole walk
    assert "covered 200 m" not in note, note


def test_the_backstop_serves_no_budget_and_no_clock():
    """CLAUDE.md law: the model never hears a budget fact.

    A served budget makes the model settle -- measured, and the reason the
    flight arm above logs its count instead of stating it. The clock arm
    inherits the same contract, so the ONLY thing a cut pass tells the model
    is which stops were not flown.
    """
    _reply, out, _flights, _asked, _rend = _explore_drive(
        [[_run('explore("prior_7")\n')], "done"],
        deadline_epoch=time.time() - 60.0,
    )
    blob = repr(out[0])
    for word in ("deadline", "budget", "wall-clock", "wall clock",
                 "seconds left", "time left", "flights remaining"):
        assert word not in blob.lower(), (word, blob)
    # no epoch-shaped number reached the model either
    assert not re.search(r"1[6-9]\d{8}", blob), blob


def test_the_flight_arm_still_fires_on_its_own():
    """The clock is a SECOND arm, not a replacement.

    Vacuity floor for this file: if the stop bound stopped working the three
    tests above would still pass, because they never let a pass reach it.
    """
    _reply, out, flights, _asked, _rend = _explore_drive(
        [[_run('explore("prior_7")\n')], "done"],
        stops=[[float(x), 200.0] for x in range(100, 100 + 40 * 5, 5)],
    )
    # flight_budget is 32 in this fixture; the walk asks for 40 stops.
    assert len(flights) == 32, len(flights)
    assert out[0]["explored"]["stops"] == 32
