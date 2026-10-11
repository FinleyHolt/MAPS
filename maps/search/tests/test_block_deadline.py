"""One block cannot hang the whole mission on a slow bounded loop."""

from __future__ import annotations

import time

from maps.search.parse_compile import python_surface as ps


def _mission_python():
    env = ps.MissionEnv(
        run_plan=lambda plan, said: {},
        rows_of_ref=lambda ref: [],
        plan_rn=[0],
        discourse={},
        minted=[],
        say_err=lambda ack, msg: {**ack, "error": msg},
        row_of_fid=lambda fid: None,
    )
    return ps.MissionPython(env)


def test_a_slow_bounded_block_stops_at_the_wall_clock_ceiling(monkeypatch):
    monkeypatch.setattr(ps, "TRACE_DEADLINE_S", 0.05)
    mp = _mission_python()
    # A bounded loop (well under the 100k LINE ceiling) whose body is slow C
    # work, so it trips the wall-clock ceiling first. Bounded so the test ends
    # in a couple of seconds even if the ceiling never fired.
    started = time.monotonic()
    out = mp.run_block(
        "total = 0\nfor i in range(20000):\n    total += sum(range(20000))\n"
    )
    elapsed = time.monotonic() - started
    assert "error" in out, out
    assert "past" in out["error"], out["error"]
    # it is the WALL-CLOCK ceiling, not the line ceiling
    assert "lines" not in out["error"], out["error"]
    # and it stopped early, not after the full 20000 iterations
    assert elapsed < 2.0, f"did not stop early: {elapsed:.2f}s"


def test_the_ceilings_are_local_to_the_block(monkeypatch):
    """The executor under a block runs UNTRACED."""
    import sys

    seen: dict = {}

    def _run_plan(plan, said):
        seen["trace"] = sys.gettrace()
        return {}

    mp = _mission_python()
    mp.env.run_plan = _run_plan
    out = mp.run_block('look_for("car")\n')
    assert "error" not in out, out
    assert "trace" in seen
    if hasattr(sys, "monitoring"):
        assert seen["trace"] is None
    # and the block's own lines are still counted
    assert mp._lines >= 1


def test_a_runaway_loop_inside_a_nested_def_still_hits_the_line_ceiling(
        monkeypatch):
    """A def, lambda or comprehension in the block is the block's own code: its lines
    count, or a runaway one line deep would run unbounded."""
    monkeypatch.setattr(ps, "TRACE_LINE_CEILING", 2_000)
    mp = _mission_python()
    out = mp.run_block("def spin():\n    n = 0\n    while True:\n"
                       "        n += 1\nspin()\n")
    assert "unbounded loop" in (out.get("error") or ""), out
