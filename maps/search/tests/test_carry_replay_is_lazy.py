"""A handle re-derives on read, and only the ops it reads are replayed.

A handle is a query: after a flight it answers about the flown world
(``test_flight_mints_a_handle`` pins that law). What this file pins is the
COST of that law. The replay used to run the mission's whole op history
against the new surface on the first read after a flight, so a flight late
in a mission paid for every compare the mission had run -- 32 s on the Cal
City surface with 60 blocks behind it. A read now replays the handle's own
dependency closure and nothing else, and the rows it hands back are the
ones a fresh computation over the same surface makes.
"""

from __future__ import annotations

import ast

from maps.search.parse_compile import reasoner_loop as rl
from maps.search.tests.test_flight_mints_a_handle import _drive, _run


def _programs_run(monkeypatch) -> list[list[dict]]:
    """Every op-program the executor runs, in order."""
    seen: list[list[dict]] = []
    orig = rl._run_program

    def _spy(cands, anchors, program, **kw):
        seen.append(list(program.get("ops") or []))
        return orig(cands, anchors, program, **kw)

    monkeypatch.setattr(rl, "_run_program", _spy)
    return seen


def _printed_fids(result: dict) -> list[str]:
    """The fid list a block printed (a small grid carries no fids array)."""
    return ast.literal_eval(str(result.get("printed") or "[]"))


def test_a_read_after_a_flight_replays_its_own_closure_and_nothing_else(
        monkeypatch):
    seen = _programs_run(monkeypatch)
    results, _graph, added = _drive([
        [_run('cars = find("car")')],
        [_run('go_to(prior_7)')],
        [_run('near = cars.filter(near=prior_7, dwithin=500)')],
        [_run('gate = find("prior_1")')],
        [_run('gate_by = gate.order(nearest=prior_7)')],   # never read again
        [_run('go_to(prior_7)')],
        [_run('print(sorted(r.fid for r in near))')],     # the read after it
        [_run('fresh = find("car").filter(near=prior_7, dwithin=500)\n'
              'print(sorted(r.fid for r in fresh))')],
        "done",
    ])
    n_flight = len(seen)
    # -- the rows: the second flight's cars are in the re-derived handle,
    # and the handle reads exactly what a fresh computation over the flown
    # surface reads (the full-history replay made the same rows; a lazy one
    # must).
    shown = _printed_fids(results[6])
    assert set(added[1]) <= set(shown), (added, shown)
    assert shown == _printed_fids(results[7]), (shown, results[7])
    assert len(shown) == len(added[0]) + len(added[1]), shown
    # -- the cost: the programs run for the read after the second flight
    # carry the chain behind ``near`` (the find, the compare, the cut) and
    # NEVER the unread ``gate``/``gate_by`` chain.
    after = [op for prog in seen[n_flight - 2:] for op in prog]
    assert any(op.get("op") == "find"
               and (op.get("args") or {}).get("query") == "car"
               for op in after), after
    assert not any((op.get("args") or {}).get("query") == "prior_1"
                   for op in after), after
    assert not any(op.get("op") == "score_by" for op in after), after


def test_a_handle_never_read_between_two_flights_is_replayed_once(
        monkeypatch):
    """The replay is charged to the READ, not to the flight: two flights and
    then one read pay for one re-derivation, not two."""
    seen = _programs_run(monkeypatch)
    results, _graph, added = _drive([
        [_run('cars = find("car")')],
        [_run('go_to(prior_7)')],
        [_run('go_to(prior_7)')],
        [_run('print(sorted(r.fid for r in cars))')],
        "done",
    ])
    shown = _printed_fids(results[3])
    assert set(added[0]) | set(added[1]) <= set(shown), (added, shown)
    finds = [op for prog in seen for op in prog if op.get("op") == "find"
             and (op.get("args") or {}).get("query") == "car"]
    # once for the block that bound it, once for the read after the flights
    assert len(finds) == 2, len(finds)


def test_a_replay_the_executor_refuses_leaves_the_handle_stale_and_unclaimed(
        monkeypatch):
    """An executor refusal on the replay is a failed replay: the handle keeps
    the rows it held and is NOT marked fresh, so the next read tries again.
    The full-history replay accepted the refusal as success and served every
    handle stale for the rest of the mission."""
    orig = rl.run_geom_program_tool
    refused = [0]

    def _refusing(args, **kw):
        ops = ((args or {}).get("program") or {}).get("ops") or []
        # a replay is the only program that re-runs the find over a
        # NON-EMPTY seed (the mission's own first find sees an empty
        # carry); refuse exactly those, twice
        if kw.get("seed_results") and any(
                op.get("op") == "find"
                and (op.get("args") or {}).get("query") == "car"
                for op in ops) and refused[0] < 2:
            refused[0] += 1
            return {"error": "cannot resolve a POINT-LIST from 'r103'"}
        return orig(args, **kw)

    monkeypatch.setattr(rl, "run_geom_program_tool", _refusing)
    results, _graph, added = _drive([
        [_run('cars = find("car")')],
        [_run('go_to(prior_7)')],
        [_run('print(sorted(r.fid for r in cars))')],   # refused: stale
        [_run('print(sorted(r.fid for r in cars))')],   # refused again
        [_run('print(sorted(r.fid for r in cars))')],   # replays: fresh
        "done",
    ])
    assert refused[0] == 2
    assert _printed_fids(results[2]) == [] and _printed_fids(results[3]) == []
    assert set(added[0]) <= set(_printed_fids(results[4]))
