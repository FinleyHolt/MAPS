"""Bench-only withholds: each takes exactly its part away, and none serves the flown
text byte for byte."""

from __future__ import annotations

import difflib
import hashlib
import json

import pytest

from maps.llm.endpoint_spec import EndpointSpec
from maps.search import withhold as W
from maps.search.parse_compile import contract, prompts
from maps.search.parse_compile import python_surface as ps
from maps.search.parse_compile import reasoner_agent as ra
from maps.search.parse_compile.code_surface import act_needs_a_thing
from maps.search.parse_compile.reasoner_loop import _PICTURE_WITHHELD_OF
from maps.search.tests.fakes import (
    quiet_env,
    cars,
    drive,
    look_drive,
    map_mission,
    run,
    tool_results,
)

FLOWN_SHA256 = {
    "oryx_orx32_25mm_nadir": "b977e7290c6652029f9d57c63c358ee670ec541549511af289357310627a0dd7",
    "nadir_1008_90": "9dfc2661077dcef616f0f78451bd08a0242098aa1d7cf8aca322b70e314471ed",
    None: "400bc4dbc4ed3459c7673e9a91ec5d3a2a39bee715a18ed6771e2245467bc024",
}


def _dump(monkeypatch, names: str, sensor: str | None = "oryx_orx32_25mm_nadir") -> str:
    monkeypatch.setenv(W.ENV, names)
    return contract.dump_text(sensor=sensor, agl_m=914.4)


def _sections(text: str) -> dict[str, str]:
    out, key = {}, None
    for ln in text.splitlines(keepends=True):
        if ln.startswith("=== ") and ln.split()[1] in ("SYSTEM", "TOOLS", "SURFACE", "USER"):
            key = ln.split()[1]
            out[key] = ""
        out[key] += ln
    return out


def _diff(a: str, b: str) -> tuple[int, int]:
    lines = list(difflib.ndiff(a.splitlines(), b.splitlines()))
    return (sum(ln.startswith("- ") for ln in lines),
            sum(ln.startswith("+ ") for ln in lines))


@pytest.mark.parametrize("sensor", list(FLOWN_SHA256))
def test_no_withhold_serves_the_flown_contract(monkeypatch, sensor):
    text = _dump(monkeypatch, "", sensor)
    assert hashlib.sha256(text.encode()).hexdigest() == FLOWN_SHA256[sensor]
    assert prompts.served_doc() == prompts.CODE_DOC


# withhold -> (sections that change, marker strings it takes away,
# (removed, added) lines of the turn, of the description, of the surface)
CASES = {
    "prior": (("SYSTEM", "TOOLS"),
              ("never a coordinate", "do not move you", "holds only PRIOR",
               "not make it movable", "the prior map alone",
               "prior (map features)", "prior footprints", "only what OSM recorded",
               "prior_3123", "nothing surveyed", "only looking will"),
              (13, 12), (7, 7), (0, 0)),
    "images": ((), (), (0, 0), (0, 0), (0, 0)),
    "look_for": (("SYSTEM", "TOOLS", "SURFACE"),
                 ("look_for", "watched classes", "segmenting your", "ARM EVERY CLASS",
                  "WHAT YOU SEE BECOMES ROWS", "The flight folds", "SAM segments roads",
                  "pools = find(", "empty SAM response", "SAM covers"),
                 (23, 10), (21, 3), (1, 0)),
    "look": (("SYSTEM", "TOOLS", "SURFACE"),
             ("look(", "a CLOSE look", "known only from a look",
              "WHAT NO COLUMN CARRIES IS IN THE PIXELS", "look at them",
              "RELATE and LOOK", "look and relate", "loop to LOOK"),
             (13, 6), (8, 5), (1, 0)),
    "mission_map": (("SYSTEM",), ("=== MISSION MAP ===", "You are the reasoner"),
                    (40, 0), (0, 0), (0, 0)),
    "end_commit": (("SYSTEM",), ("Your mission ends", "commits nothing",
                                 "A reply before any flight", "Your answer is the ROW"),
                   (11, 3), (0, 0), (0, 0)),
    "loop": (("TOOLS",), ("cannot go inside a loop", "explore(the set)"),
             (0, 0), (1, 1), (0, 0)),
    "doc": (("TOOLS",), ("Write ordinary Python", "THE MAP -- free", "ARM EVERY CLASS",
                         "WHAT YOU SEE BECOMES ROWS", "ORDINARY PYTHON RUNS HERE"),
            (0, 0), (76, 14), (0, 0)),
    "map": (("SYSTEM", "TOOLS", "SURFACE"),
            ("show_map", "crops and maps", "the distinction the map cannot make",
             "THE MAP -- free"),
            (5, 4), (5, 4), (1, 0)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_each_withhold_takes_exactly_its_part(monkeypatch, name):
    changes, markers, turn, doc, surface = CASES[name]
    base = _sections(_dump(monkeypatch, ""))
    got = _sections(_dump(monkeypatch, name))
    for key in base:
        if key in changes:
            assert got[key] != base[key], key
        else:
            # an unchanged section keeps its bytes, length header included
            assert got[key] == base[key], key
    assert _diff(prompts.CODE_SYSTEM_PROMPT, prompts.served_system_prompt()) == turn
    assert _diff(prompts.CODE_DOC, prompts.served_doc()) == doc
    base_surface = base["SURFACE"].split("\n", 1)[1]
    got_surface = got["SURFACE"].split("\n", 1)[1]
    assert _diff(base_surface, got_surface) == surface
    served = (prompts.served_system_prompt() + prompts.served_doc()
              + "\n".join(contract.surface_lines()) + got["SYSTEM"])
    for m in markers:
        assert m in base["SYSTEM"] + base["TOOLS"] + base_surface, m
        assert m not in served, m


def test_an_end_commit_withhold_keeps_the_cue_word_standard_as_one_bullet(monkeypatch):
    monkeypatch.setenv(W.ENV, "end_commit")
    assert ("- When you know which object the instruction means -- and a cue word no\n"
            "  column carries is known only from a look, never from rank -- go_to()\n"
            "  its row, then reply.\n") in prompts.served_system_prompt()


def test_a_look_for_withhold_serves_no_segmenter_in_the_sensor_block(monkeypatch):
    base = _sections(_dump(monkeypatch, "", "nadir_1008_90"))["SYSTEM"]
    got = _sections(_dump(monkeypatch, "look_for", "nadir_1008_90"))["SYSTEM"]
    assert "SAM may merge" in base and "small entity counts ±1" in got
    assert "SAM" not in got, got


def test_the_system_turn_withhold_serves_no_system_message(monkeypatch):
    monkeypatch.setenv(W.ENV, "system_turn")
    assert prompts.served_system_prompt("RGB, nadir.") == ""
    spec = EndpointSpec(base_url="http://unused", model_id="inkling-small", max_model_len=262144,
                        tokenizer_name="qwen", max_output_tokens=40000)
    r = ra.Reasoner(endpoint_spec=spec,
                    sensor_description="RGB, nadir.")
    assert r.messages == []
    got = _sections(_dump(monkeypatch, "system_turn"))
    base = _sections(_dump(monkeypatch, ""))
    assert got["SYSTEM"] == "=== SYSTEM (0 chars) ===\n\n\n"
    assert got["TOOLS"] == base["TOOLS"] and got["SURFACE"] == base["SURFACE"]


def test_withheld_verbs_are_unbound(monkeypatch):
    monkeypatch.setenv(W.ENV, "look_for,look")
    assert "look_for" not in ps.served_calls() and "look" not in ps.served_calls()
    mp = ps.MissionPython(quiet_env())
    assert "look_for" not in mp.ns and "look" not in mp.ns and "go_to" in mp.ns
    out = json.dumps(mp.run_block('look_for("pool")'))
    assert "did you mean look_for" not in out


def test_a_look_for_withhold_takes_its_remedies(monkeypatch):
    assert "look_for" in act_needs_a_thing(name="go_to", got="1, 2")
    monkeypatch.setenv(W.ENV, "look_for")
    assert "look_for" not in act_needs_a_thing(name="go_to", got="1, 2")


def test_an_images_withhold_delivers_the_looks_caption(ortho_dir, monkeypatch):
    monkeypatch.setenv(W.ENV, "images")
    results, reqs, _recs, vision = look_drive(
        [[run('a = find("car")')], [run("look(a)")], "over"], cars(3), ortho_dir)
    assert reqs, "the look rendered"
    assert len(vision) == 1
    assert _PICTURE_WITHHELD_OF.format("crop") in json.dumps(results)


def test_an_unknown_withhold_is_refused(monkeypatch):
    monkeypatch.setenv(W.ENV, "pictures")
    with pytest.raises(ValueError, match="unknown withholds"):
        W.active()


def _map_mission(script, monkeypatch, names="", renderer=None, finds=1):
    pytest.importorskip("cv2")
    monkeypatch.setenv(W.ENV, names)
    return map_mission(script, renderer=renderer, finds=finds)


def _crops(reqs):
    return [{"jpeg": b"x", "center_fid": r.get("center_fid") or "", "legend": [],
             "context_m": 30.0, "world_window": [0, 0, 60, 60], "marks_dropped": 0,
             "marks_out_of_view": [], "marks_unbadged_in_frame": []} for r in reqs]


def test_a_look_withhold_map_caption_says_fly_alone(monkeypatch):
    _, vision = _map_mission(["show_map(around=prior_7)"], monkeypatch)
    assert "fly and look: the camera" in vision[0][0]
    _, vision = _map_mission(["show_map(around=prior_7)"], monkeypatch, "look")
    assert "fly: the camera" in vision[0][0] and "fly and look" not in vision[0][0]


_PICTURE_CALLS = {"show_map(around=prior_7)": "World-state maps",
                  "go_to(prior_7)": "Overhead crops",
                  "a = explore(prior_1)": "The pass's catch",
                  "look(prior_7)": "Overhead crops"}


@pytest.mark.parametrize("call", sorted(_PICTURE_CALLS))
def test_an_images_withhold_serves_each_picture_as_its_caption(call, monkeypatch):
    _, vision = _map_mission([call], monkeypatch, renderer=_crops)
    assert [len(j) for _, j in vision] == [1]
    _, vision = _map_mission([call], monkeypatch, "images", renderer=_crops)
    assert [len(j) for _, j in vision] == [0], vision
    assert vision[0][0].startswith(_PICTURE_CALLS[call])


@pytest.mark.parametrize("call", ["show_map(around=prior_7)", "look(prior_7)",
                                  "a = explore(prior_1)"])
def test_an_images_withhold_counts_no_picture_delivered(call, monkeypatch):
    results, _ = _map_mission([call], monkeypatch, "images", renderer=_crops)
    served = json.dumps(results)
    assert '"delivered": 0' in served and '"delivered": 1' not in served, served


def test_an_images_withhold_flight_view_note_says_its_picture_is_not_served(monkeypatch):
    results, _ = _map_mission(["go_to(prior_7)"], monkeypatch, "images", renderer=_crops)
    stop = results[0]["flown"]["flown"][0]
    assert stop["note"] == _PICTURE_WITHHELD_OF.format("view"), stop


def test_an_images_withhold_keeps_the_catchs_text_facts(monkeypatch):
    """Only the picture is withheld: the catch keeps every other key."""
    def _catch(names):
        results, _ = _map_mission(["a = explore(prior_1)"], monkeypatch, names,
                                  renderer=_crops)
        return next(r["explored"]["catch"] for r in results if "explored" in r)

    base, held = _catch(""), _catch("images")
    assert set(held) == set(base) | {"note"}, (base, held)
    assert held["note"] == _PICTURE_WITHHELD_OF.format("crop"), held


def test_a_look_for_withhold_serves_no_watchlist(monkeypatch):
    script = ["go_to(prior_7)", "a = explore(prior_1)", "show_map(around=prior_7)"]
    for finds in (1, 0):
        results, _ = _map_mission(script, monkeypatch, renderer=_crops, finds=finds)
        assert "watching" in json.dumps(results)
        results, _ = _map_mission(script, monkeypatch, "look_for", renderer=_crops,
                                  finds=finds)
        served = json.dumps(results)
        assert all("error" not in r for r in results), served
        assert "watching" not in served and "look_for" not in served, served


def test_a_prior_withhold_flies_to_a_point_and_serves_no_prior_row(monkeypatch):
    script = [[run('go_to([30, 30])\nexplore([40, 40])\ngo_to("[50, 50]")\n')], "done"]
    _reply, fr, flights, _d = drive(script, cands=[])
    assert "takes a THING" in (tool_results(fr)[0].get("error") or "")
    assert not flights
    monkeypatch.setenv(W.ENV, "prior")
    _reply, fr, flights, _d = drive(script, cands=[])
    served = json.dumps(tool_results(fr))
    assert "error" not in served and "prior_" not in served, served
    assert [f["target_fid"] for f in flights] == ["[30, 30]", "[40, 40]", "[50, 50]"], flights


def test_the_served_literals_hold_their_worked_block_and_every_edit_target(monkeypatch):
    """The worked block in CODE_DOC scans on the real surface and runs the
    chain it teaches; every withhold edit names text its literal carries once,
    and a combination's edit names text its singles serve once."""
    import ast

    worked = "\n".join(
        ln[2:] for ln in prompts.CODE_DOC.partition("This is one block:\n")[2]
        .partition("\nEach line runs")[0].splitlines() if ln.startswith("  "))
    assert worked.count("\n") >= 5, worked
    tree = ps.scan_block(worked)
    calls = [n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert {"look_for", "find", "look", "explore", "go_to"} <= set(calls), calls
    assert calls.index("explore") < calls.index("go_to"), calls
    assert "find" in calls[calls.index("explore"):], calls
    find_kwargs = {k.arg for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "find" for k in n.keywords}
    assert "color" in find_kwargs, find_kwargs
    assert find_kwargs & {"nearest", "near", "within", "east_of", "west_of",
                          "north_of", "south_of"}, find_kwargs

    from maps.perception.geometry.sensor_spec import resolve_sensor

    for edits, literal, turn in ((prompts._DOC_EDITS, prompts.CODE_DOC, False),
                                 (prompts._TURN_EDITS, prompts.CODE_SYSTEM_PROMPT, True),
                                 (prompts._SENSOR_EDITS, resolve_sensor("nadir_1008_90").description,
                                  False)):
        singles = {k: v for k, v in edits.items() if "+" not in k}
        for name, items in edits.items():
            assert set(name.split("+")) <= W.WITHHOLDS.keys(), name
            monkeypatch.setenv(W.ENV, name.replace("+", ","))
            text = prompts._served(literal, singles, turn) if "+" in name else literal
            for e in items:
                assert prompts._edit(text, e, turn) not in (None, text), (name, e)
                units = prompts._units(text, turn)
                if e[0] == "drop":
                    n = sum(u.startswith(e[1]) for u in units)
                elif e[0] == "sentence" or (e[0] == "sub" and turn):
                    n = sum(e[1] in (prompts._flat(u) if turn else u) for u in units)
                else:
                    n = 1 if e[0] == "cut" else text.count(e[1])
                assert n == 1, (name, e)


def test_a_prior_withhold_empty_find_names_no_prior(monkeypatch):
    from maps.search.analyze_scene.geom_program import Result, compact_results

    def hint():
        return compact_results({"r0": Result(kind="miss")})["r0"]["hint"]

    assert "searched every surface row (prior + runtime) and" in hint()
    monkeypatch.setenv(W.ENV, "prior")
    assert "searched every surface row and matched nothing" in hint()
    assert "prior" not in hint()


def test_a_prior_withhold_map_caption_and_color_note_name_no_survey(monkeypatch):
    from maps.search.parse_compile.carry import _preflight_color_note
    from maps.search.parse_compile.views import map_vision_text

    assert "NNNN chip" in map_vision_text([]) and "priors" in _preflight_color_note()
    monkeypatch.setenv(W.ENV, "prior")
    caption = map_vision_text([])
    for word in ("survey", "NNNN", "roads", "hatching", "landuse"):
        assert word not in caption, (word, caption)
    assert "detected_N" in caption and "no prior map" in caption
    assert "prior" not in _preflight_color_note().replace("no prior map", "")


def test_a_prior_withhold_empty_cut_and_semicolon_refusal_name_no_survey(monkeypatch):
    from maps.search.analyze_scene import geom_program as gp
    from maps.search.analyze_scene.geom_program import ProgramError

    road = {"fid": "prior_R", "xy": [0.0, 0.0], "name": "Orchid Drive",
            "mask": [[[-5.0, -500.0], [5.0, -500.0], [5.0, 500.0], [-5.0, 500.0]]],
            "centerline": None, "cells": [], "search_text": "orchid drive"}
    west = [{"fid": "w", "xy": [-300.0, 10.0], "class": "car",
             "mask": [[[-305.0, 5.0], [-295.0, 5.0], [-295.0, 15.0], [-305.0, 15.0]]],
             "centerline": None, "cells": [], "search_text": "car"}]
    cmp = {"ref": "r1", "op": "compare", "args": {
        "items": ["w"], "refs": [{"ref": "prior_R", "as": "orchid_drive"}],
        "cardinals": [{"dir": "east", "ref": "orchid_drive"}]}}
    side = {"ref": "r2", "op": "filter_items", "args": {
        "items": "r1", "key": "east of orchid_drive", "op": "on_side"}}
    facing = {"ref": "r2", "op": "filter_items", "args": {
        "items": "r1", "key": "facing orchid_drive", "op": "in_or_blank",
        "value": ["west"]}}

    def served():
        stats = [gp.run_program(west, [road], {"ops": [cmp, cut]})["results"]["r2"].cut_stats
                 for cut in (side, facing)]
        with pytest.raises(ProgramError) as e:
            gp.run_program(west, [road], {"ops": [{"ref": "r1", "op": "find", "args": {
                "query": "cuisine=pizza;kebab"}}]})
        return json.dumps(stats) + str(e.value)

    flown = served()
    assert "prior map" in flown and "surveyed" in flown and "OSM" in flown, flown
    monkeypatch.setenv(W.ENV, "prior")
    got = served()
    assert "nothing_on_that_side" in got and "nothing_reads_those_words" in got, got
    for word in ("prior", "surveyed", "OSM"):
        assert word not in got, (word, got)


def test_a_look_withhold_coordinate_refusal_names_no_looking(monkeypatch):
    from maps.search.parse_compile.code_surface import act_needs_a_thing

    assert "Points are for RELATING and LOOKING --" in act_needs_a_thing(name="go_to", got="1, 2")
    monkeypatch.setenv(W.ENV, "look")
    text = act_needs_a_thing(name="go_to", got="1, 2")
    assert "Points are for RELATING --" in text and "LOOKING" not in text, text


def test_a_look_withhold_loop_move_refusal_names_no_looking(monkeypatch):
    def refusal():
        with pytest.raises(ps.CodeError) as e:
            ps.scan_block("for r in find('car'):\n    go_to(r)\n")
        return str(e.value)

    assert "Loop to LOOK, then move once" in refusal()
    monkeypatch.setenv(W.ENV, "look")
    text = refusal()
    assert "Loop to read the rows, then move once" in text and "LOOK" not in text, text


def test_a_loop_withhold_lets_a_move_ride_a_loop(monkeypatch):
    block = "for r in find('car'):\n    go_to(r)\nf = go_to\n"
    with pytest.raises(ps.CodeError, match="cannot go inside a loop"):
        ps.scan_block(block)
    monkeypatch.setenv(W.ENV, "loop")
    ps.scan_block(block)


def test_a_loop_withhold_ends_the_mission_on_its_first_text_turn(monkeypatch):
    script = ["", [run('go_to("prior_7")')], "done"]
    _reply, fr, flights, _d = drive(script)
    assert flights and any("[Repair]" in str(m["content"]) for m in fr.messages)
    monkeypatch.setenv(W.ENV, "loop")
    reply, fr, flights, _d = drive(script)
    assert reply.ended_by == "reply" and reply.turns_used == 1 and not flights
    assert not any("[Repair]" in str(m["content"]) for m in fr.messages)


def test_a_loop_withhold_reruns_an_identical_program(monkeypatch):
    script = [[run('a = find("car")\n')], [run('b = find("car")\n')], "done"]
    _reply, fr, _f, _d = drive(script)
    assert "no new information" in json.dumps(tool_results(fr)[1])
    monkeypatch.setenv(W.ENV, "loop")
    _reply, fr, _f, _d = drive(script)
    second = tool_results(fr)[1]
    assert "no new information" not in json.dumps(second) and "car_1" in json.dumps(second)


def test_a_loop_withhold_flies_a_point_as_asked(monkeypatch):
    script = [[run("go_to([3, 4])\n")], "done"]
    monkeypatch.setenv(W.ENV, "prior")
    _reply, fr, flights, _d = drive(script)
    assert flights == [{"target_fid": "prior_7"}], flights
    assert "own position" in json.dumps(tool_results(fr))
    monkeypatch.setenv(W.ENV, "prior,loop")
    _reply, fr, flights, _d = drive(script)
    assert flights == [{"target_fid": "[3, 4]"}], flights
    assert "own position" not in json.dumps(tool_results(fr))


def test_a_loop_withhold_replays_no_reasoning(monkeypatch):
    from maps.search.tests.fakes import TEST_SPEC

    def _fake(url, api_key, **kwargs):
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": "done", "reasoning_content": "I weighed two rooftops."}}]}

    monkeypatch.setattr(ra, "chat_completions_raw", _fake)
    for names, replayed in (("", "I weighed two rooftops."), ("loop", "")):
        monkeypatch.setenv(W.ENV, names)
        r = ra.Reasoner(endpoint_spec=TEST_SPEC)
        out = r.step_tool_call([])
        assert out["reasoning"] == "I weighed two rooftops."
        assert r.messages[-1]["reasoning_content"] == replayed


def test_the_text_family_serves_no_turn_and_the_calls_signatures(monkeypatch):
    """doc + system_turn is the research text family: an empty system turn and
    a description that is the surface's one-line signatures."""
    got = _sections(_dump(monkeypatch, "doc,system_turn"))
    assert got["SYSTEM"] == "=== SYSTEM (0 chars) ===\n\n\n"
    lines = contract.surface_lines()
    assert prompts.served_doc() == "\n".join(lines) + "\n"
    assert lines[0].startswith("  find(") and lines[-1] == "  <rows>.distance(other)"


def test_a_map_withhold_unbinds_show_map_and_its_remedies(monkeypatch):
    stance = [[run('stand(at="prior_7")\n')], "done"]
    assert "show_map(around=)" in act_needs_a_thing(name="go_to", got="1, 2")
    assert "crops and maps render" in json.dumps(tool_results(drive(stance)[1]))
    monkeypatch.setenv(W.ENV, "map")
    assert "show_map" not in ps.served_calls()
    assert "show_map" not in ps.MissionPython(quiet_env()).ns
    text = act_needs_a_thing(name="go_to", got="1, 2")
    assert "show_map" not in text and "near=, within=, stand(at=)." in text, text
    served = json.dumps(tool_results(drive(stance)[1]))
    assert "crops render its facing up" in served and "maps" not in served, served
