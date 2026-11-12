"""A handle re-derives on read, and only the ops it reads are replayed."""

from __future__ import annotations

import ast

from maps.library import result_handles as rl
from maps.mission.tests.fakes import grow_drive, run


def _programs_run(monkeypatch) -> list[list[dict]]:
    """Every op-program the executor runs, in order."""
    seen: list[list[dict]] = []
    orig = rl.run_program

    def _spy(cands, anchors, program, **kw):
        seen.append(list(program.get("ops") or []))
        return orig(cands, anchors, program, **kw)

    monkeypatch.setattr(rl, "run_program", _spy)
    return seen


def _printed_fids(result: dict) -> list[str]:
    """The fid list a block printed (a small grid carries no fids array)."""
    return ast.literal_eval(str(result.get("printed") or "[]"))


def test_a_read_after_a_flight_replays_its_own_closure_and_nothing_else(
        monkeypatch):
    seen = _programs_run(monkeypatch)
    results, _graph, added = grow_drive([
        [run('cars = find("car")')],
        [run('go_to(prior_7)')],
        [run('near = cars.filter(near=prior_7, dwithin=500)')],
        [run('gate = find("prior_1")')],
        [run('gate_by = gate.order(nearest=prior_7)')],   # never read again
        [run('go_to(prior_7)')],
        [run('print(sorted(r.fid for r in near))')],     # the read after it
        [run('fresh = find("car").filter(near=prior_7, dwithin=500)\n'
              'print(sorted(r.fid for r in fresh))')],
        "done",
    ])
    n_flight = len(seen)
    # -- the rows: the second flight's cars are in the re-derived handle,
    # and the handle reads exactly what a fresh computation over the flown
    # surface reads.
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
    """The replay is charged to the READ, not to the flight: two flights and then one
    read pay for one re-derivation, not two."""
    seen = _programs_run(monkeypatch)
    results, _graph, added = grow_drive([
        [run('cars = find("car")')],
        [run('go_to(prior_7)')],
        [run('go_to(prior_7)')],
        [run('print(sorted(r.fid for r in cars))')],
        "done",
    ])
    shown = _printed_fids(results[3])
    assert set(added[0]) | set(added[1]) <= set(shown), (added, shown)
    finds = [op for prog in seen for op in prog if op.get("op") == "find"
             and (op.get("args") or {}).get("query") == "car"]
    # once for the block that bound it, once for the read after the flights
    assert len(finds) == 2, len(finds)


def test_a_replay_the_executor_refuses_fails_the_read_and_the_next_read_retries(
        monkeypatch):
    """An executor refusal on the replay fails the read that asked, naming the
    handle; it is NOT marked fresh, so the next read tries again."""
    from maps.query.op_io import ProgramError

    orig = rl.run_program
    refused = [0]

    def _refusing(cands, anchors, program, **kw):
        # a replay is the only program that re-runs the find over a
        # NON-EMPTY seed (the mission's own first find sees an empty
        # carry); refuse exactly those, twice
        if kw.get("seed_results") and any(
                op.get("op") == "find"
                and (op.get("args") or {}).get("query") == "car"
                for op in program["ops"]) and refused[0] < 2:
            refused[0] += 1
            raise ProgramError("cannot resolve a POINT-LIST from 'r103'")
        return orig(cands, anchors, program, **kw)

    monkeypatch.setattr(rl, "run_program", _refusing)
    results, _graph, added = grow_drive([
        [run('cars = find("car")')],
        [run('go_to(prior_7)')],
        [run('print(sorted(r.fid for r in cars))')],   # refused
        [run('print(sorted(r.fid for r in cars))')],   # refused again
        [run('print(sorted(r.fid for r in cars))')],   # replays: fresh
        "done",
    ])
    assert refused[0] == 2
    for res in results[2:4]:
        assert not res.get("printed"), res
        assert "cars cannot be re-derived on the surface as it stands: " \
            "cannot resolve a POINT-LIST" in str(res.get("error")), res
    assert set(added[0]) <= set(_printed_fids(results[4]))


def test_the_first_read_after_a_flight_sees_the_flown_rows_whoever_reads_it():
    """look() reads a handle the way print/len do: the first read after a flight
    re-derives it, so the crops cover the cars the flight added."""
    results, _graph, added = grow_drive([
        [run('cars = find("car")')],
        [run('f = go_to(prior_7)')],     # nothing in this block reads a handle
        [run('look(cars)')],
        "done",
    ])
    assert results[2].get("crop", {}).get("looked") == len(added[0]), results[2]


def test_a_refused_re_derive_fails_the_read_naming_the_handle():
    """near's union over cars passes at 201 cars and is refused at 402: the
    read after the second flight is refused by name, never served the 201
    rows near held before it."""
    res, _g, _a = grow_drive([[run('go_to("prior_7")\n')],
                              [run('cars = find("car")\nnear = cars.order(nearest=cars)\nlen(near)\n')],
                              [run('go_to("prior_1")\n')],
                              [run('len(cars)\nlen(near)\n')], "done"], per_flight=201)
    last = res[-1]
    assert "len(cars) = 402" in last.get("printed", "")
    assert "len(near) = 201" not in last.get("printed", ""), last
    assert "near cannot be re-derived" in str(last.get("error")), last


def test_a_refused_re_derive_names_only_the_handles_whose_replay_hit_it():
    """A program that reads two handles, one of which re-derives fine, is
    refused naming only the one whose replay the executor refused."""
    res, _g, _a = grow_drive([[run('go_to("prior_7")\n')],
                              [run('cars = find("car")\nnear = cars.order(nearest=cars)\n')],
                              [run('go_to("prior_1")\n')],
                              [run('near.filter(near=cars, dwithin=5)\n')], "done"],
                             per_flight=201)
    err = str(res[-1].get("error"))
    assert err.startswith("near cannot be re-derived"), err
